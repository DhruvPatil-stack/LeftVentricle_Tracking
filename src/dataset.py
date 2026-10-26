"""EchoNet-Dynamic cache builder: stream videos from the zip and save labeled frames + masks."""
from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.draw import polygon as sk_polygon


ZIP_PATH = Path(__file__).resolve().parents[1] / "EchoNet-Dynamic.zip"
ZIP_ROOT = "EchoNet-Dynamic"
FRAME_SIZE = (112, 112)


def load_csvs(zip_path: Path = ZIP_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FileList and VolumeTracings from the EchoNet zip."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"{ZIP_ROOT}/FileList.csv") as f:
            file_list = pd.read_csv(f)
        with z.open(f"{ZIP_ROOT}/VolumeTracings.csv") as f:
            tracings = pd.read_csv(f)
    return file_list, tracings


def build_mask(points: np.ndarray, shape: tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    """Build a filled LV mask from 20 short-axis chord endpoints.

    The polygon starts at the X1,Y1 endpoints in order, then returns along
    the X2,Y2 endpoints in reverse order. The first row of the frame (long axis) is skipped by the caller.
    """
    xs = np.concatenate([points[:, 0], points[::-1, 2]])
    ys = np.concatenate([points[:, 1], points[::-1, 3]])
    rr, cc = sk_polygon(ys, xs, shape=shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rr, cc] = 255
    return mask


def group_tracings(tracings: pd.DataFrame) -> dict[str, dict[int, np.ndarray]]:
    """Group tracings by FileName then Frame; skip the first (long-axis) row of each frame."""
    cols = tracings[["X1", "Y1", "X2", "Y2"]].to_numpy()
    out: dict[str, dict[int, np.ndarray]] = {}
    for (name, frame_index), group in tracings.groupby(["FileName", "Frame"], sort=False):
        values = cols[group.index.to_numpy()]
        if len(values) < 2:
            continue
        out.setdefault(name, {})[int(frame_index)] = values[1:]
    return out


def read_frames(video_path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Return grayscale frames at the requested indices in one sequential pass."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    out: dict[int, np.ndarray] = {}
    target_max = max(wanted) if wanted else -1
    index = 0
    while index <= target_max:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if index in wanted:
            out[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        index += 1
    cap.release()
    return out


def extract_video(zip_handle: zipfile.ZipFile, filename_avi: str, out_path: Path) -> bool:
    """Write a single video from the zip to disk; return False if it is missing."""
    name = f"{ZIP_ROOT}/Videos/{filename_avi}"
    try:
        info = zip_handle.getinfo(name)
    except KeyError:
        return False
    with zip_handle.open(info) as source, open(out_path, "wb") as sink:
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            sink.write(chunk)
    return True


def cache_one_video(
    zip_handle: zipfile.ZipFile,
    file_name: str,
    tracings_lookup: dict[int, np.ndarray],
    cache_dir: Path,
    split: str,
    ef: float,
    esv: float,
    edv: float,
    tmp_dir: Path,
) -> tuple[bool, str]:
    """Cache labeled frames and masks for one video; returns (success, reason_if_skipped)."""
    filename_avi = f"{file_name}.avi"
    if not tracings_lookup:
        return False, "no tracings"
    tmp_video = tmp_dir / f"{file_name}.avi"
    if not extract_video(zip_handle, filename_avi, tmp_video):
        return False, "video missing in zip"
    try:
        wanted = {int(k) for k in tracings_lookup.keys()}
        gray_by_index = read_frames(tmp_video, wanted)
        frames, masks, frame_indices = [], [], []
        for frame_index, chords in sorted(tracings_lookup.items()):
            gray = gray_by_index.get(int(frame_index))
            if gray is None or gray.shape != FRAME_SIZE:
                continue
            frames.append(gray)
            masks.append(build_mask(chords))
            frame_indices.append(int(frame_index))
        if len(frames) < 2:
            return False, "fewer than 2 labeled frames"
        out_path = cache_dir / f"{file_name}.npz"
        np.savez_compressed(
            out_path,
            frames=np.stack(frames).astype(np.uint8),
            masks=np.stack(masks).astype(np.uint8),
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            split=np.asarray(split),
            ef=np.asarray(ef, dtype=np.float32),
            esv=np.asarray(esv, dtype=np.float32),
            edv=np.asarray(edv, dtype=np.float32),
        )
        return True, "ok"
    finally:
        try:
            tmp_video.unlink()
        except FileNotFoundError:
            pass


def iter_build_cache(
    splits: tuple[str, ...] = ("TRAIN", "VAL"),
    cache_dir: Path | None = None,
    limit: int | None = None,
) -> dict:
    """Build the cache for the given splits. Returns a summary dict."""
    cache_dir = cache_dir or (Path(__file__).resolve().parents[1] / "data" / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_list, tracings = load_csvs()
    tracings_grouped = group_tracings(tracings)
    selected = file_list[file_list["Split"].isin(splits)].reset_index(drop=True)
    if limit is not None:
        selected = selected.head(limit)
    saved, skipped, reasons = 0, 0, {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(ZIP_PATH) as zh:
            for row in selected.itertuples(index=False):
                avi = f"{row.FileName}.avi"
                ok, reason = cache_one_video(
                    zh,
                    row.FileName,
                    tracings_grouped.get(avi, {}),
                    cache_dir,
                    row.Split,
                    float(row.EF),
                    float(row.ESV),
                    float(row.EDV),
                    tmp_dir,
                )
                if ok:
                    saved += 1
                else:
                    skipped += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
    return {"saved": saved, "skipped": skipped, "reasons": reasons}
