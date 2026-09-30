import cv2
import numpy as np


def get_lv_contour(mask: np.ndarray) -> np.ndarray | None:
    """
    Extract the largest contour from a binary LV mask.
    The LV cavity is assumed to be the single largest connected region.
    Returns the contour array or None if no contour is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def track_contours(masks: list[np.ndarray]) -> list[np.ndarray | None]:
    """Return the dominant LV contour for each mask in the sequence."""
    return [get_lv_contour(m) for m in masks]


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
