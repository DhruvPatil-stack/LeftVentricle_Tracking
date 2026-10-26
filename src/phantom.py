"""Labeled synthetic ultrasound phantom for code validation."""
from __future__ import annotations
import cv2
import numpy as np


def half_ellipsoid_volume(a: float, b: float) -> float:
    """Return half-ellipsoid volume in pixel cubed."""
    return float((2.0 / 3.0) * np.pi * a * b * b)


def make_phantom(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Make a multi-beat LV phantom with masks and exact volume."""
    rng = np.random.default_rng(seed)
    fps, beats, frames_per_beat, size = 50, int(rng.integers(3, 6)), 50, 112
    total = beats * frames_per_beat
    a0, b0 = rng.uniform(30, 38), rng.uniform(18, 25)
    sl, sr = rng.uniform(0.05, 0.15), rng.uniform(0.10, 0.45)
    masks = np.zeros((total, size, size), dtype=np.uint8)
    frames, volumes = np.zeros_like(masks), []
    yy, xx = np.mgrid[:size, :size]
    for index in range(total):
        phase = index % frames_per_beat / frames_per_beat
        contraction = (1.0 - np.cos(2.0 * np.pi * phase)) / 2.0
        a, b = a0 * (1.0 - sl * contraction), b0 * (1.0 - sr * contraction)
        shift_x, shift_y = rng.integers(-2, 3, size=2)
        cx, apex = size // 2 + shift_x, 24 + shift_y
        cavity = (((xx - cx) / b) ** 2 + ((yy - (apex + a)) / a) ** 2 <= 1) & (yy >= apex)
        masks[index, cavity] = 255
        myocardium = cv2.dilate(masks[index], np.ones((5, 5), np.uint8)) > 0
        image = np.full((size, size), 60.0)
        image[myocardium] = 180.0
        image[cavity] = 25.0
        decoy = ((xx - 87) / 18) ** 2 + ((yy - 35) / 20) ** 2 <= 1
        image[decoy] = 80.0
        speckle = rng.rayleigh(0.35, (size, size))
        image = cv2.GaussianBlur(image * (0.7 + speckle), (3, 3), 0)
        frames[index] = np.clip(image + rng.normal(0, 4), 0, 255).astype(np.uint8)
        volumes.append(half_ellipsoid_volume(a, b))
    volumes = np.asarray(volumes)
    true_ef = 100.0 * (1.0 - volumes.min() / volumes.max())
    return frames, masks, volumes, float(true_ef)
