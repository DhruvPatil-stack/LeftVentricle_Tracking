import cv2
import numpy as np
import pytest
from src.contour_tracking import get_lv_contour, contour_mask
from src.ef_calculation import compute_lv_areas, estimate_ejection_fraction, method_of_disks
from src.phantom import half_ellipsoid_volume, make_phantom


def test_area_ratio_ef_is_sixty_percent():
    result = estimate_ejection_fraction([1000.0, 800.0, 400.0], fps=50)
    assert result["area_ratio"]["median_pct"] == pytest.approx(60.0)


def test_ellipse_contour_area_is_close():
    mask = np.zeros((240, 240), np.uint8)
    cv2.ellipse(mask, (120, 120), (50, 30), 0, 0, 360, 255, -1)
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert compute_lv_areas([contour[0]])[0] == pytest.approx(np.pi * 50 * 30, rel=0.01)


def test_disk_volume_is_close_for_ellipse():
    a, b = 80, 40
    mask = np.zeros((240, 240), np.uint8)
    cv2.ellipse(mask, (120, 120), (a, b), 0, 0, 360, 255, -1)
    expected = 4.0 / 3.0 * np.pi * a * b * b
    assert method_of_disks(mask) == pytest.approx(expected, rel=0.03)


def test_overlap_tracking_rejects_larger_decoy():
    first = np.zeros((100, 100), np.uint8); cv2.circle(first, (30, 50), 15, 255, -1)
    previous = get_lv_contour(first)
    second = np.zeros((100, 100), np.uint8); cv2.circle(second, (33, 50), 15, 255, -1); cv2.circle(second, (75, 50), 22, 255, -1)
    chosen = get_lv_contour(second, previous)
    assert contour_mask(chosen, second.shape)[50, 33] == 255
    assert contour_mask(chosen, second.shape)[50, 75] == 0


def test_empty_masks_and_short_video():
    assert compute_lv_areas([None, None]) == [0.0, 0.0]
    with pytest.raises(ValueError): estimate_ejection_fraction([0.0, 0.0], fps=50)


def test_phantom_volume_formula():
    frames, masks, volumes, ef = make_phantom(42)
    assert frames.dtype == np.uint8 and masks.shape == frames.shape
    assert half_ellipsoid_volume(3, 2) == pytest.approx((2.0 / 3.0) * np.pi * 3 * 4)
    assert 0 < ef < 100
