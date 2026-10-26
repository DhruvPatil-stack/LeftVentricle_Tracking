"""Area and mask-based research estimators for LV ejection fraction."""
from __future__ import annotations
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter


def compute_lv_areas(contours: list[np.ndarray | None]) -> list[float]:
    """Measure each contour area in pixel squared."""
    return [0.0 if c is None else float(cv2.contourArea(c)) for c in contours]


def smooth_area_curve(areas: list[float], fps: float, seconds: float = 0.1, order: int = 2) -> np.ndarray:
    """Smooth approximately 0.1 seconds of area samples."""
    values = np.asarray(areas, dtype=float)
    if len(values) < 5:
        return values
    window = max(5, int(round(fps * seconds)))
    window += window % 2 == 0
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    return savgol_filter(values, window, min(order, window - 1))


def find_ed_es_pairs(
    areas: list[float],
    fps: float,
    min_beat_seconds: float = 0.4,
    min_prominence_fraction: float = 0.3,
) -> tuple[list[tuple[int, int]], bool, np.ndarray, int]:
    """Find ED peaks and the following ES valley for every complete beat.

    ES frames with a zero raw area (a failed mask) are excluded as candidates.
    Only accept ED candidates whose prominence is at least a fraction of the full
    area swing (A_max - A_min); the swing scales with the true beat amplitude,
    so this rejects plateau and trough bumps that are not real diastole.
    Returns (pairs, used_single_pair_fallback, smooth, rejected_peak_count).
    """
    smooth = smooth_area_curve(areas, fps)
    if not len(smooth) or np.max(smooth) <= 0:
        raise ValueError("No valid LV area exists.")
    raw = np.asarray(areas, dtype=float)
    valid_frame = raw > 0
    distance = max(1, int(round(fps * min_beat_seconds)))
    # Prominence floor scales with the beat swing so small bumps do not qualify as ED.
    swing = float(smooth.max() - smooth.min())
    ed_all, _ = find_peaks(smooth, distance=distance)
    if min_prominence_fraction > 0 and swing > 0:
        ed_strong, _ = find_peaks(smooth, distance=distance, prominence=min_prominence_fraction * swing)
    else:
        ed_strong = ed_all
    rejected = int(len(ed_all) - len(ed_strong))
    pairs: list[tuple[int, int]] = []
    for first, second in zip(ed_strong[:-1], ed_strong[1:]):
        window = np.arange(int(first) + 1, int(second))
        valid = window[valid_frame[window]]
        if len(valid):
            es_index = int(valid[np.argmin(smooth[valid])])
            pairs.append((int(first), es_index))
    if pairs:
        return pairs, False, smooth, rejected
    ed_index = int(np.argmax(smooth))
    valid_full = np.where(valid_frame)[0]
    if len(valid_full) == 0:
        raise ValueError("No valid LV area exists.")
    es_index = int(valid_full[np.argmin(smooth[valid_full])])
    return [(ed_index, es_index)], True, smooth, rejected


def method_of_disks(mask: np.ndarray, disks: int = 20) -> float:
    """Estimate volume with midpoint-rule disk diameters along the PCA long axis."""
    points = np.argwhere(mask > 0)
    if len(points) < 3:
        return 0.0
    xy = points[:, ::-1].astype(float)
    center = xy.mean(axis=0)
    _, _, vectors = np.linalg.svd(xy - center, full_matrices=False)
    long_axis = vectors[0]
    normal_axis = np.array([-long_axis[1], long_axis[0]])
    long_coords, short_coords = (xy - center) @ long_axis, (xy - center) @ normal_axis
    start, end = long_coords.min(), long_coords.max()
    length = end - start
    if length <= 0:
        return 0.0
    step = length / disks
    midpoints = start + (np.arange(disks) + 0.5) * step
    diameters = []
    for mid in midpoints:
        band = short_coords[np.abs(long_coords - mid) <= 0.5]
        if len(band) == 0:
            band = short_coords[np.abs(long_coords - mid) <= 1.0]
        diameters.append(0.0 if len(band) == 0 else float(band.max() - band.min()))
    return float((np.pi / 4.0) * step * np.sum(np.square(diameters)))


def _ef(ed: float, es: float) -> float:
    """Calculate percentage change from ED to ES."""
    return float("nan") if ed <= 0 else 100.0 * (1.0 - es / ed)


def _summary(values: list[float]) -> dict:
    """Summarize one estimator across beats."""
    array = np.asarray(values, dtype=float)
    return {"per_beat_pct": [round(float(v), 4) for v in array], "median_pct": None if not len(array) else round(float(np.nanmedian(array)), 4), "std_pct": None if not len(array) else round(float(np.nanstd(array)), 4)}


def estimate_ejection_fraction(
    areas: list[float],
    masks: list[np.ndarray] | None = None,
    fps: float = 50.0,
    min_beat_seconds: float = 0.4,
    min_prominence_fraction: float = 0.3,
) -> dict:
    """Report per-beat EF estimates and quality flags."""
    pairs, fallback, smooth, rejected = find_ed_es_pairs(areas, fps, min_beat_seconds, min_prominence_fraction)
    area_values, power_values, disk_values = [], [], []
    for ed, es in pairs:
        area_values.append(_ef(areas[ed], areas[es]))
        power_values.append(100.0 * (1.0 - (areas[es] / areas[ed]) ** 1.5) if areas[ed] > 0 else float("nan"))
        if masks is not None:
            disk_values.append(_ef(method_of_disks(masks[ed]), method_of_disks(masks[es])))
    values = np.asarray(areas, dtype=float)
    jumps = np.abs(np.diff(values)) / np.maximum(values[:-1], 1.0) if len(values) > 1 else np.array([])
    return {"pairs": [{"ed_frame_index": ed, "es_frame_index": es} for ed, es in pairs], "used_single_pair_fallback": fallback, "area_ratio": _summary(area_values), "area_power_volume_proxy": _summary(power_values), "method_of_disks": _summary(disk_values), "quality_flags": {"empty_mask_fraction": round(float(np.mean(values == 0)), 4), "large_area_jump_fraction": round(float(np.mean(jumps > 0.5)) if len(jumps) else 0.0, 4), "number_of_beats": len(pairs), "rejected_ed_peaks": int(rejected)}, "smoothed_areas": [round(float(value), 4) for value in smooth]}


def plot_area_curve(areas: list[float], result: dict, save_path: str | None = None) -> None:
    """Plot raw and smoothed areas with ED and ES markers."""
    frames = np.arange(len(areas))
    plt.figure(figsize=(10, 4))
    plt.plot(frames, areas, label="Raw LV area", color="lightsteelblue")
    plt.plot(frames, result["smoothed_areas"], label="Smoothed LV area", color="steelblue")
    for pair in result["pairs"]:
        plt.axvline(pair["ed_frame_index"], color="green", linestyle="--")
        plt.axvline(pair["es_frame_index"], color="red", linestyle="--")
    plt.xlabel("Frame"); plt.ylabel("Area (pixel squared)"); plt.title("Left-ventricle cavity area")
    plt.legend(); plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()
