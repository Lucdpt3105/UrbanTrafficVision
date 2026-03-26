import argparse
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ── Face detection ────────────────────────────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils.classes import (
    COCO_CLASSES, VEHICLE_CLASSES, OBSTACLE_CLASSES,
    TRAFFIC_LIGHT_CLASS, PERSON_CLASS,
    SPECIAL_LABEL_BY_RATIO, VEHICLE_SUBTYPES,
)
from utils.colors import COLORS
from utils.draw import (
    draw_box, draw_fps, draw_dashboard,
    draw_driver_cam_border, draw_traffic_light_indicator,
)
from utils.traffic_light import detect_traffic_light_state

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def classify_vehicle_detail(coco_class: str, roi_bgr: np.ndarray,
                             box_w: int, box_h: int) -> str:
    if box_h == 0:
        return coco_class.capitalize()

    ratio = box_w / box_h  # >1 = wide, <1 = tall

    # ── aspect-ratio heuristic mapping ────────────────────────────────────────
    if coco_class in SPECIAL_LABEL_BY_RATIO:
        thresholds = SPECIAL_LABEL_BY_RATIO[coco_class]   # [(min_ratio, label), …]
        # thresholds ordered descending
        for min_ratio, label in thresholds:
            if ratio >= min_ratio:
                return label

    # ── colour-based refinements for cars ────────────────────────────────────
    if coco_class == "car" and roi_bgr is not None and roi_bgr.size > 0:
        try:
            small = cv2.resize(roi_bgr, (60, 40))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            sat_mean = float(np.mean(hsv[:, :, 1]))
            val_mean = float(np.mean(hsv[:, :, 2]))

            if ratio > 1.5 and val_mean > 160:
                return "SUV"
            if ratio > 1.3 and sat_mean < 40:
                return "Sedan"
            if ratio < 0.85:
                return "Minivan"
        except Exception:
            pass

    # ── motorbike splitting ───────────────────────────────────────────────────
    if coco_class == "motorcycle":
        if box_h > 80 and ratio < 1.1:
            return "Motorcycle"
        return "Scooter"

    # ── default: first entry in VEHICLE_SUBTYPES ─────────────────────────────
    sub_list = VEHICLE_SUBTYPES.get(coco_class, [])
    return sub_list[0] if sub_list else coco_class.capitalize()



def find_driver_cam_region(frame: np.ndarray):
    """
    Find driver camera region by detecting faces in corners.
    More reliable than brightness detection.
    """
    h, w = frame.shape[:2]
    cam_w = w // 4
    cam_h = h // 4

    corners = {
        "top-left":     (0,          0,          cam_w,     cam_h),
        "top-right":    (w - cam_w,  0,          w,         cam_h),
        "bottom-left":  (0,          h - cam_h,  cam_w,     h),
        "bottom-right": (w - cam_w,  h - cam_h,  w,         h),
    }

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_corner = None
    best_score = 0

    for name, (x1, y1, x2, y2) in corners.items():
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            continue

        # Detect faces in this region
        faces = FACE_CASCADE.detectMultiScale(
            region, 
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            maxSize=(cam_w // 2, cam_h // 2)
        )

        # If faces found, this is likely the driver cam
        if len(faces) > 0:
            # Score = number of faces detected
            if len(faces) > best_score:
                best_score = len(faces)
                best_corner = (x1, y1, x2, y2)

    # Fallback: if no faces detected, use brightness (for non-camera regions)
    if best_corner is None:
        main_brightness = float(np.mean(gray))
        for name, (x1, y1, x2, y2) in corners.items():
            region = gray[y1:y2, x1:x2]
            if region.size == 0:
                continue
            brightness = float(np.mean(region))
            brightness_diff = abs(brightness - main_brightness)
            if brightness_diff > 15.0:
                best_corner = (x1, y1, x2, y2)
                break

def load_yolo(tiny: bool, conf_thresh: float, nms_thresh: float):
    """Load YOLOv8 model (nano or medium)."""
    model_name = "yolov8n.pt" if tiny else "yolov8m.pt"
    print(f"[INFO] Loading {model_name} …", end=" ", flush=True)
    
    try:
        model = YOLO(model_name)
        # Try to use CUDA if available
        try:
            device = 0  # GPU device 0
            model.to(device)
            print("(CUDA enabled) done.")
        except:
            device = "cpu"
            model.to(device)
            print("done.")
        
        # Store confidence and NMS for later use
        model.conf = conf_thresh
        model.iou = nms_thresh
        return model
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        sys.exit(1)


def run_yolo(model, frame: np.ndarray, conf_thresh: float) -> list:
    """Run YOLOv8 inference on frame."""
    h, w = frame.shape[:2]
    
    # Run inference
    results = model(frame, conf=conf_thresh, verbose=False, iou=model.iou)
    
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cid = int(box.cls[0])
            
            # Map YOLO class ID to COCO class name
            # YOLOv8 uses COCO classes by default
            name = result.names[cid] if cid < len(result.names) else "unknown"
            
            detections.append({
                "class_id": cid,
                "class_name": name,
                "conf": conf,
                "x1": max(0, x1),
                "y1": max(0, y1),
                "x2": min(w - 1, x2),
                "y2": min(h - 1, y2),
            })
    
    return detections

def _safe_roi(frame, x1, y1, x2, y2):
    """Clip coordinates and return ROI (may be empty)."""
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def categorise(det: dict, frame: np.ndarray):
    
    name = det["class_name"]
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    roi = _safe_roi(frame, x1, y1, x2, y2)

    if name == PERSON_CLASS:
        return {**det, "category": "person", "label": "Person", "roi": roi}

    if name == TRAFFIC_LIGHT_CLASS:
        state = detect_traffic_light_state(roi)
        return {**det, "category": "traffic_light",
                "label": f"Traffic Light {state}", "roi": roi, "tl_state": state}

    if name in VEHICLE_CLASSES:
        detail = classify_vehicle_detail(
            name, roi, x2 - x1, y2 - y1)
        return {**det, "category": "vehicle", "label": detail, "roi": roi}

    if name in OBSTACLE_CLASSES:
        return {**det, "category": "obstacle",
                "label": name.replace("_", " ").title(), "roi": roi}

    # Everything else → generic obstacle
    return {**det, "category": "obstacle",
            "label": name.replace("_", " ").title(), "roi": roi}


def label_driver_cam_persons(detections: list, driver_cam_box) -> list:
    """
    For each 'person' detection whose bbox overlaps with the driver-cam region,
    relabel them as 'Driver' and mark is_driver_cam=True.
    """
    if driver_cam_box is None:
        return detections

    dx1, dy1, dx2, dy2 = driver_cam_box
    relabelled = []
    for det in detections:
        d = dict(det)
        if d["category"] == "person":
            # Overlap check
            ox1 = max(d["x1"], dx1); oy1 = max(d["y1"], dy1)
            ox2 = min(d["x2"], dx2); oy2 = min(d["y2"], dy2)
            if ox2 > ox1 and oy2 > oy1:      # overlap exists
                d["label"] = "Driver"
                d["is_driver_cam"] = True
            else:
                d["is_driver_cam"] = False
        else:
            d["is_driver_cam"] = False
        relabelled.append(d)
    return relabelled


WINDOW_TITLE = "DNN Vehicle Detection — Press Q to quit"

def process_video(args):
    # ── Load model ─────────────────────────────────────────────────────────
    model = load_yolo(args.tiny, args.conf, args.nms)

    # ── Create display window ───────────────────────────────────────────────
    if args.show:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    # ── Open video ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.input}")
        sys.exit(1)

    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Input : {args.input}  ({vid_w}×{vid_h}  {fps_vid:.1f}fps  {total} frames)")
    if args.scale != 1.0:
        print(f"[INFO] Processing at {args.scale*100:.0f}% scale for faster inference")
    print(f"[INFO] Driver camera detection: {'enabled (Face Detection)' if args.detect_driver else 'disabled'}")

    # ── Output writer ───────────────────────────────────────────────────────
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_vid, (vid_w, vid_h))
        print(f"[INFO] Output: {args.output}")

    # ── State ───────────────────────────────────────────────────────────────
    driver_cam_region  = None           # (x1,y1,x2,y2) or None
    driver_cam_checked = False
    frame_idx          = 0
    fps_display        = 0.0
    t_prev             = time.time()

    print("[INFO] Processing … (press Q to quit if display window is open)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── FPS measurement ──────────────────────────────────────────────
        t_now = time.time()
        elapsed = t_now - t_prev
        if elapsed > 0:
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / elapsed)
        t_prev = t_now

        # ── Scale down for faster processing ─────────────────────────────
        if args.scale != 1.0:
            h_scaled = int(frame.shape[0] * args.scale)
            w_scaled = int(frame.shape[1] * args.scale)
            frame_proc = cv2.resize(frame, (w_scaled, h_scaled))
        else:
            frame_proc = frame

        # ── Driver-cam detection (less frequent: every 120 frames) ─────────
        if args.detect_driver and (not driver_cam_checked or frame_idx % 120 == 0):
            driver_cam_region = find_driver_cam_region(frame_proc)
            # Scale back if needed
            if driver_cam_region and args.scale != 1.0:
                x1, y1, x2, y2 = driver_cam_region
                driver_cam_region = (
                    int(x1 / args.scale),
                    int(y1 / args.scale),
                    int(x2 / args.scale),
                    int(y2 / args.scale)
                )
            driver_cam_checked = True

        # ── YOLO inference ───────────────────────────────────────────────
        raw_dets = run_yolo(model, frame_proc, args.conf)

        # ── Scale detections back if needed ──────────────────────────────
        if args.scale != 1.0:
            for det in raw_dets:
                scale_factor = 1.0 / args.scale
                det["x1"] = int(det["x1"] * scale_factor)
                det["y1"] = int(det["y1"] * scale_factor)
                det["x2"] = int(det["x2"] * scale_factor)
                det["y2"] = int(det["y2"] * scale_factor)

        # ── Categorise ──────────────────────────────────────────────────
        detections = [categorise(d, frame) for d in raw_dets]

        # ── Tag driver-cam persons ───────────────────────────────────────
        detections = label_driver_cam_persons(detections, driver_cam_region)

        # ── Collect stats ────────────────────────────────────────────────
        counts: dict = {}
        tl_state = None

        for det in detections:
            lbl = det["label"]
            cat = det["category"]

            if cat == "traffic_light":
                tl_state = det.get("tl_state", "UNKNOWN")
                key = f"Traffic Light ({tl_state})"
            elif cat == "person":
                key = det["label"]          # "Driver" or "Person"
            elif cat == "vehicle":
                key = lbl
            else:
                key = lbl

            counts[key] = counts.get(key, 0) + 1

        # ── Draw driver-cam border ───────────────────────────────────────
        if driver_cam_region is not None:
            draw_driver_cam_border(frame, *driver_cam_region)

        # ── Draw detections ──────────────────────────────────────────────
        for det in detections:
            draw_box(
                frame,
                det["x1"], det["y1"], det["x2"], det["y2"],
                category=det["category"],
                label=det["label"],
                conf=det["conf"],
                is_driver_cam=det.get("is_driver_cam", False),
            )
            # Extra traffic light widget
            if det["category"] == "traffic_light":
                state = det.get("tl_state", "UNKNOWN")
                draw_traffic_light_indicator(
                    frame,
                    det["x2"] + 4,
                    max(0, det["y1"]),
                    state,
                )

        # ── Overlays ─────────────────────────────────────────────────────
        draw_dashboard(frame, counts, tl_state)
        draw_fps(frame, fps_display)

        # Progress print every 30 frames
        if frame_idx % 30 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"\r  Frame {frame_idx}/{total} ({pct:.1f}%)  FPS:{fps_display:.1f}  "
                  f"Detections:{len(detections)}   ", end="", flush=True)

        # ── Write / show ─────────────────────────────────────────────────
        if writer:
            writer.write(frame)

        if args.show:
            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[INFO] Quit by user.")
                break

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"\n[INFO] Saved → {args.output}")
    cv2.destroyAllWindows()
    print("\n[INFO] Done.")

def parse_args():
    p = argparse.ArgumentParser(
        description="DNN-based traffic & vehicle detection (YOLOv8 + OpenCV)")
    p.add_argument("--input",  "-i", required=True,
                   help="Path to input video file or camera index (0, 1, …)")
    p.add_argument("--output", "-o",
                   default=os.path.join(OUTPUT_DIR, "output_video.mp4"),
                   help="Path to write output video (default: output/output_video.mp4)")
    p.add_argument("--conf",   "-c", type=float, default=0.25,
                   help="Confidence threshold [0-1] (default: 0.25)")
    p.add_argument("--nms",    "-n", type=float, default=0.45,
                   help="NMS IoU threshold [0-1] (default: 0.45)")
    p.add_argument("--nano", dest="tiny", action="store_true", default=True,
                   help="Use YOLOv8n (nano, faster, default)")
    p.add_argument("--medium", dest="tiny", action="store_false",
                   help="Use YOLOv8m (medium, more accurate)")
    p.add_argument("--show", dest="show", action="store_true", default=True,
                   help="Enable live preview window (default: True)")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="Disable live preview window (useful on headless servers)")
    p.add_argument("--no-output", dest="output", action="store_const",
                   const=None,
                   help="Disable video output (display only)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale factor for input video (0.5 = half resolution, faster) (default: 1.0)")
    p.add_argument("--no-driver-cam", dest="detect_driver", action="store_false", default=True,
                   help="Disable driver camera detection (faster, no CAM label)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args)
