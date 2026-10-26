"""ED/ES pairing: multi-beat videos, failed-mask ES, prominence filter."""
import numpy as np
from src.ef_calculation import find_ed_es_pairs, estimate_ejection_fraction


def _sin_beats(fps: float = 50.0, beats: int = 4, frames_per_beat: int = 40, peak: float = 1000.0, trough: float = 400.0) -> list[float]:
    """Synthetic area curve with several clear heartbeats."""
    length = beats * frames_per_beat
    amplitude = (peak - trough) / 2.0
    offset = (peak + trough) / 2.0
    t = np.arange(length)
    phase = 2.0 * np.pi * t / frames_per_beat
    curve = offset + amplitude * np.cos(phase)
    return [float(x) for x in curve]


def test_multi_beat_pairing_finds_several_pairs():
    """A 5-beat area curve must yield at least 3 ED-ES pairs."""
    areas = _sin_beats(beats=5)
    pairs, fallback, smooth, _ = find_ed_es_pairs(areas, fps=50.0)
    assert not fallback
    assert len(pairs) >= 3
    for ed, es in pairs:
        assert ed < es
        assert smooth[ed] > smooth[es]


def test_failed_mask_cannot_become_es():
    """A failed mask (area == 0) must not be chosen as the ES frame."""
    areas = _sin_beats(beats=3, frames_per_beat=40)
    inject_at = 60
    areas[inject_at] = 0.0
    pairs, fallback, _, _ = find_ed_es_pairs(areas, fps=50.0)
    for _, es in pairs:
        assert areas[es] > 0.0, f"ES fell on failed mask frame {es}"
    result = estimate_ejection_fraction(areas, masks=None, fps=50.0)
    assert all(p["es_frame_index"] != inject_at for p in result["pairs"])


def test_plateau_bump_is_rejected_as_ed():
    """A diastolic plateau bump after each ES must not become an ED candidate.

    Synthetic 5-beat cosine with a cleanly-separated plateau bump added to the
    diastolic rise of each beat. Five cosine cycles give four interior ED peaks
    (t=0 is dropped by find_peaks as an endpoint). Four interior ED peaks give
    3 ED-ES pairs. Each bump is 80 units — well below f*swing = 0.3*600 = 180 —
    so the prominence filter rejects every bump but keeps every true ED.
    Each FAC must be within 1 pp of (peak-trough)/peak = 60.
    """
    frames_per_beat = 40
    base = _sin_beats(beats=5, frames_per_beat=frames_per_beat)
    areas = list(base)
    # Inject a wide, moderate-height plateau bump between ED and the next trough.
    # Wide enough (5 samples) to survive Savitzky-Golay smoothing with window 5,
    # moderate enough (90 units) to stay below f*swing = 0.3*600 = 180.
    for b in range(5):
        center = b * frames_per_beat + 10  # middle of diastolic plateau to next trough
        for off in range(-2, 3):
            idx = center + off
            if 0 <= idx < len(areas):
                areas[idx] = areas[idx] + 90.0
    pairs_no_prom, _, _, _ = find_ed_es_pairs(areas, fps=50.0, min_prominence_fraction=0.0)
    pairs_prom, _, _, rejected = find_ed_es_pairs(areas, fps=50.0, min_prominence_fraction=0.3)
    assert len(pairs_no_prom) > len(pairs_prom), "prominence filter must drop the diastolic bumps"
    assert rejected >= 1
    result = estimate_ejection_fraction(areas, masks=None, fps=50.0)
    assert len(result["pairs"]) == 3, f"expected 3 pairs, got {len(result['pairs'])}"
    for fac in result["area_ratio"]["per_beat_pct"]:
        assert abs(fac - 60.0) <= 1.0, f"FAC {fac} differs from truth 60.0 by more than 1 pp"


def test_trough_bump_does_not_produce_degenerate_pair():
    """A small bump inside the systolic trough must not create an ED about equal to ES."""
    base = _sin_beats(beats=3, frames_per_beat=40, peak=1000.0, trough=400.0)
    areas = list(base)
    trough_index = 20  # ES of first beat
    areas[trough_index] = areas[trough_index] + 40.0  # small bump inside the trough
    result = estimate_ejection_fraction(areas, masks=None, fps=50.0)
    for pair, fac in zip(result["pairs"], result["area_ratio"]["per_beat_pct"]):
        ed_area = areas[pair["ed_frame_index"]]
        es_area = areas[pair["es_frame_index"]]
        # Degenerate pair would show ED ~ ES; a real beat shows ED materially above ES.
        assert ed_area - es_area > 100.0, f"ED {ed_area} too close to ES {es_area}"
        assert fac > 20.0, f"FAC {fac} is near 0% — likely a degenerate pair"
