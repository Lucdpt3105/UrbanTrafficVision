"""
Traffic light state detection using HSV color analysis.
"""
import cv2
import numpy as np

# HSV ranges for traffic light colors
_RED_LOWER1  = np.array([0,   120,  70])
_RED_UPPER1  = np.array([10,  255, 255])
_RED_LOWER2  = np.array([170, 120,  70])
_RED_UPPER2  = np.array([180, 255, 255])
_YELLOW_LOWER = np.array([18,  100, 100])
_YELLOW_UPPER = np.array([35,  255, 255])
_GREEN_LOWER  = np.array([36,   60,  60])
_GREEN_UPPER  = np.array([89,  255, 255])


def detect_traffic_light_state(roi_bgr: np.ndarray) -> str:
    """Detect traffic light state using optimized HSV analysis."""
    if roi_bgr is None or roi_bgr.size == 0:
        return "UNKNOWN"

    h, w = roi_bgr.shape[:2]
    if h < 6 or w < 3:
        return "UNKNOWN"

    # Smaller patch resize for faster processing
    patch = cv2.resize(roi_bgr, (15, 45), interpolation=cv2.INTER_LINEAR)
    hsv   = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # Divide into regions
    ph = hsv.shape[0]
    third = max(1, ph // 3)
    top_third = hsv[:third]
    mid_third = hsv[third: 2 * third]
    bot_third = hsv[2 * third:]

    def lit_pixels(region, lower, upper, lower2=None, upper2=None):
        if region.size == 0:
            return 0
        mask = cv2.inRange(region, lower, upper)
        if lower2 is not None:
            mask |= cv2.inRange(region, lower2, upper2)
        return int(np.sum(mask > 0))

    scores = {
        "RED":    lit_pixels(top_third, _RED_LOWER1, _RED_UPPER1, _RED_LOWER2, _RED_UPPER2),
        "YELLOW": lit_pixels(mid_third, _YELLOW_LOWER, _YELLOW_UPPER),
        "GREEN":  lit_pixels(bot_third, _GREEN_LOWER,  _GREEN_UPPER),
    }

    best = max(scores, key=scores.get)
    if scores[best] < 2:  # Lower threshold
        return "UNKNOWN"

    return best
