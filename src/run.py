"""Command-line runner for one echocardiogram video."""
import argparse
import json
from pathlib import Path
import cv2
from .config import PipelineConfig, set_seed
from .contour_tracking import build_visualisation, track_contours
from .ef_calculation import compute_lv_areas, estimate_ejection_fraction, plot_area_curve
from .extract_frames import extract_frames, get_video_metadata
from .preprocess import preprocess_frames
from .segment import TARGET_SIZE, load_model, segment_frames


def read_sidecar(weights_path: Path, config: PipelineConfig) -> tuple[tuple[int, ...], tuple[int, int]]:
    """Read features and target_size from a sibling .json next to the weights."""
    sidecar = weights_path.with_suffix(".json")
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        features = tuple(data.get("features", config.model_widths))
        size = data.get("target_size", list(TARGET_SIZE))
        return features, (int(size[0]), int(size[1]))
    return config.model_widths, TARGET_SIZE


def main() -> None:
    """Run inference and save derived outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = PipelineConfig(denoise=args.denoise)
    set_seed(config.seed)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    metadata = get_video_metadata(args.video)
    frames = extract_frames(args.video)
    processed = preprocess_frames(frames, denoise=config.denoise)
    features, target_size = read_sidecar(Path(args.weights), config)
    model = load_model(args.weights, device=args.device, widths=features)
    masks = segment_frames(model, processed, config.threshold, args.device, config.batch_size, target_size)
    contours = track_contours(masks)
    result = estimate_ejection_fraction(compute_lv_areas(contours), masks, metadata["fps"], config.min_beat_seconds)
    result["metadata"] = metadata
    result["config"] = {"seed": config.seed, "denoise": config.denoise, "threshold": config.threshold, "features": list(features), "target_size": list(target_size)}
    plot_area_curve(compute_lv_areas(contours), result, str(output / "area_curve.png"))
    writer = cv2.VideoWriter(str(output / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), metadata["fps"], (metadata["width"], metadata["height"]))
    for frame in build_visualisation(frames, contours): writer.write(frame)
    writer.release()
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
