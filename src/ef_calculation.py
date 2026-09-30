import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_lv_areas(contours: list[np.ndarray | None]) -> list[float]:
    """
    Compute 2D pixel area for each LV contour.
    Frames with no detected contour are assigned area 0.0.
    """
    areas = []
    for contour in contours:
        if contour is None:
            areas.append(0.0)
        else:
            areas.append(cv2.contourArea(contour))
    return areas


def find_ed_es_frames(areas: list[float]) -> tuple[int, int]:
    """
    Identify end-diastolic (ED) and end-systolic (ES) frame indices.
    ED = frame with maximum LV area (fully filled ventricle).
    ES = frame with minimum non-zero LV area (fully contracted ventricle).
    """
    area_array = np.array(areas)
    ed_frame = int(np.argmax(area_array))

    non_zero = area_array[area_array > 0]
    if non_zero.size == 0:
        raise ValueError("No valid LV contours found across the sequence.")

    min_nonzero_val = non_zero.min()
    es_frame = int(np.where(area_array == min_nonzero_val)[0][0])

    return ed_frame, es_frame


def estimate_ejection_fraction(areas: list[float]) -> dict:
    """
    Estimate EF using the simplified area-based proxy:
        EF (%) = (EDV - ESV) / EDV * 100
    where EDV and ESV are approximated by the ED and ES 2D pixel areas.
    This is a 2D proxy; a calibrated pixel-to-cm² conversion improves accuracy.
    """
    ed_frame, es_frame = find_ed_es_frames(areas)
    edv = areas[ed_frame]
    esv = areas[es_frame]

    if edv == 0:
        raise ValueError("End-diastolic area is zero; cannot compute EF.")

    ef = (edv - esv) / edv * 100.0

    return {
        "ejection_fraction_pct": round(ef, 2),
        "ed_frame_index": ed_frame,
        "es_frame_index": es_frame,
        "ed_area_px": round(edv, 2),
        "es_area_px": round(esv, 2),
    }


def plot_area_curve(
    areas: list[float],
    ed_frame: int,
    es_frame: int,
    save_path: str | None = None,
) -> None:
    """Plot the LV area-vs-frame curve with ED and ES markers."""
    frames = list(range(len(areas)))

    plt.figure(figsize=(10, 4))
    plt.plot(frames, areas, label="LV Area (px²)", color="steelblue", linewidth=1.5)
    plt.axvline(ed_frame, color="green", linestyle="--", label=f"ED (frame {ed_frame})")
    plt.axvline(es_frame, color="red", linestyle="--", label=f"ES (frame {es_frame})")
    plt.xlabel("Frame Index")
    plt.ylabel("LV Pixel Area (px²)")
    plt.title("Left Ventricle Area Over Cardiac Cycle")
    plt.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close()
