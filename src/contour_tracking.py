import cv2
import numpy as np


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes so cavity area is stable."""
    binary = (mask > 0).astype(np.uint8) * 255
    flood = binary.copy()
    h, w = binary.shape[:2]
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(binary, cv2.bitwise_not(flood))


def contour_mask(contour: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize one contour for overlap measurement."""
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(canvas, [contour], -1, 255, thickness=cv2.FILLED)
    return canvas


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Return binary-mask intersection over union."""
    first_bool, second_bool = first > 0, second > 0
    union = np.logical_or(first_bool, second_bool).sum()
    return 0.0 if union == 0 else float(np.logical_and(first_bool, second_bool).sum() / union)


def get_lv_contour(mask: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray | None:
    """
    Extract the largest contour from a binary LV mask.
    The LV cavity is assumed to be the single largest connected region.
    Returns the contour array or None if no contour is found.
    """
    filled = fill_holes(mask)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    if previous is None:
        return max(contours, key=cv2.contourArea)
    shape = filled.shape[:2]
    previous_mask = contour_mask(previous, shape)
    return max(contours, key=lambda contour: mask_iou(contour_mask(contour, shape), previous_mask))


def track_contours(masks: list[np.ndarray]) -> list[np.ndarray | None]:
    """Return the dominant LV contour for each mask in the sequence."""
    tracked: list[np.ndarray | None] = []
    previous = None
    for mask in masks:
        contour = get_lv_contour(mask, previous)
        tracked.append(contour)
        if contour is not None:
            previous = contour
    return tracked


def draw_contours_on_frame(
    frame: np.ndarray,
    contour: np.ndarray | None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Overlay a contour on a BGR frame for visualisation."""
    vis = frame.copy()
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, color, thickness)
    return vis


def build_visualisation(
    original_frames: list[np.ndarray],
    contours: list[np.ndarray | None],
) -> list[np.ndarray]:
    """Return a list of frames with LV contours overlaid."""
    return [draw_contours_on_frame(f, c) for f, c in zip(original_frames, contours)]
