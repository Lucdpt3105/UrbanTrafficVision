import cv2
import numpy as np

from utils.colors import COLORS, TRAFFIC_STATE_COLORS, VEHICLE_SUBTYPE_COLORS

# ── fonts ─────────────────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICK = 1
LABEL_PAD  = 4


# ── helpers ───────────────────────────────────────────────────────────────────
def _get_color(category: str, subtype: str | None = None) -> tuple:
    if subtype and subtype in VEHICLE_SUBTYPE_COLORS:
        return VEHICLE_SUBTYPE_COLORS[subtype]
    return COLORS.get(category, COLORS["default"])


def _label_size(text: str):
    (w, h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    return w, h, baseline


# ── bounding box ──────────────────────────────────────────────────────────────
def draw_box(frame: np.ndarray,
             x1: int, y1: int, x2: int, y2: int,
             category: str,
             label: str,
             conf: float,
             is_driver_cam: bool = False):
    """
    Draw a styled bounding box with label and confidence score.
    OPTIMIZED: Reduced expensive overlay operations.
    """
    color = _get_color(category, label if category == "vehicle" else None)
    if category == "traffic_light":
        state = label.split()[-1] if " " in label else "UNKNOWN"
        color = TRAFFIC_STATE_COLORS.get(state, COLORS["traffic_light"])

    # Draw border (simple, no overlay)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ── simplified label ──────────────────────────────────────────────────────
    conf_str = f"{conf * 100:.0f}%"
    full_label = f" {label}  {conf_str} "
    if is_driver_cam:
        full_label = f" CAM {label}  {conf_str} "

    lw, lh, _ = _label_size(full_label)
    lx1, ly1 = x1, max(0, y1 - lh - LABEL_PAD * 2 - 2)
    lx2, ly2 = x1 + lw + LABEL_PAD, y1

    # Simple solid background (no overlay/blend)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.putText(frame, full_label, (lx1 + LABEL_PAD, ly2 - 3),
                FONT, FONT_SCALE, (10, 10, 10), FONT_THICK, cv2.LINE_AA)


# ── driver cam region overlay ─────────────────────────────────────────────────
def draw_driver_cam_border(frame: np.ndarray,
                            x1: int, y1: int, x2: int, y2: int):
    """Draw a distinctive purple glow border around the driver-cam zone (optimized)."""
    color = (220, 80, 255)
    # Simple line, no overlay blending
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    # Label
    cv2.putText(frame, " CAM ", (x1 + 4, y1 + 20), FONT_SMALL, 0.55,
                (220, 80, 255), 1, cv2.LINE_AA)


# ── traffic light indicator ───────────────────────────────────────────────────
def draw_traffic_light_indicator(frame: np.ndarray,
                                  x: int, y: int,
                                  state: str):
    """Draw a small vertical traffic-light widget at position (x, y)."""
    w, h = 22, 60
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80),  1)

    bulbs = [
        (y + 10,  "RED"),
        (y + 30, "YELLOW"),
        (y + 50,  "GREEN"),
    ]
    for by, bstate in bulbs:
        lit = (state == bstate)
        fill_color = TRAFFIC_STATE_COLORS[bstate] if lit else (40, 40, 40)
        cv2.circle(frame, (x + w // 2, by), 7, fill_color, -1)
        if lit:
            # Glow ring
            cv2.circle(frame, (x + w // 2, by), 9, fill_color, 1)


# ── FPS overlay ───────────────────────────────────────────────────────────────
def draw_fps(frame: np.ndarray, fps: float):
    text = f" FPS: {fps:.1f} "
    (tw, th), _ = cv2.getTextSize(text, FONT_SMALL, 0.6, 1)
    h, w = frame.shape[:2]
    x, y = w - tw - 12, 10
    # Simple solid background (no expensive overlay)
    cv2.rectangle(frame, (x - 4, y), (x + tw + 4, y + th + 8), (20, 20, 20), -1)
    cv2.rectangle(frame, (x - 4, y), (x + tw + 4, y + th + 8), (60, 60, 60), 1)
    cv2.putText(frame, text, (x, y + th + 2), FONT_SMALL, 0.6,
                (0, 255, 180), 1, cv2.LINE_AA)


# ── stats dashboard ───────────────────────────────────────────────────────────
def draw_dashboard(frame: np.ndarray, counts: dict, traffic_state: str | None):
    """
    Draw a semi-transparent stats panel on the left side showing:
    - Count per detected category/subtype
    - Current traffic light state
    """
    entries = [(k, v) for k, v in sorted(counts.items()) if v > 0]
    if not entries and traffic_state is None:
        return

    line_h = 22
    padding = 8
    panel_w = 210
    rows = len(entries) + (1 if traffic_state else 0) + 1   # +1 header
    panel_h = rows * line_h + padding * 2

    h, w = frame.shape[:2]
    px, py = 8, 8

    # Simple background (no overlay blend - just solid rect)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h),
                  (15, 15, 15), -1)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h),
                  (60, 60, 60), 1)

    # Header
    cv2.putText(frame, "  DETECTION STATS", (px + 4, py + line_h),
                FONT_SMALL, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (px + 4, py + line_h + 4),
             (px + panel_w - 4, py + line_h + 4), (60, 60, 60), 1)

    row = 2
    for label, count in entries:
        # Determine color
        for cat in ["vehicle", "person", "traffic_light", "obstacle"]:
            color = COLORS.get(cat, COLORS["default"])
            if cat == "vehicle" and label in VEHICLE_SUBTYPE_COLORS:
                color = VEHICLE_SUBTYPE_COLORS[label]
                break
            if cat == "person" and label in ("Driver", "Pedestrian"):
                color = COLORS["person"]
                break
            if cat == "traffic_light" and "Light" in label:
                color = COLORS["traffic_light"]
                break
            if cat == "obstacle":
                color = COLORS["obstacle"]
                break
        txt = f"  {label[:22]:<22} {count:>3}"
        cy = py + row * line_h
        cv2.putText(frame, txt, (px + 4, cy),
                    FONT_SMALL, 0.42, color, 1, cv2.LINE_AA)
        row += 1

    # Traffic light state
    if traffic_state:
        col = TRAFFIC_STATE_COLORS.get(traffic_state, (150, 150, 150))
        txt = f"  Traffic: {traffic_state}"
        cv2.putText(frame, txt, (px + 4, py + row * line_h),
                    FONT_SMALL, 0.45, col, 1, cv2.LINE_AA)
