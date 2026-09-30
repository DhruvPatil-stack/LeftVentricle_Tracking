import cv2
import numpy as np


def preprocess_frame(
    frame: np.ndarray,
    median_ksize: int = 5,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75,
    bilateral_sigma_space: float = 75,
) -> np.ndarray:
    """
    Reduce ultrasound speckle noise while preserving LV wall edges.
    Median blur removes salt-and-pepper noise; bilateral filter smooths
    homogeneous regions without blurring the myocardium boundary.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median_filtered = cv2.medianBlur(gray, median_ksize)
    bilateral_filtered = cv2.bilateralFilter(
        median_filtered, bilateral_d, bilateral_sigma_color, bilateral_sigma_space
    )
    return bilateral_filtered


def preprocess_frames(frames: list[np.ndarray], **kwargs) -> list[np.ndarray]:
    """Apply preprocess_frame to every frame in a sequence."""
    return [preprocess_frame(f, **kwargs) for f in frames]
