"""Build the frame + mask cache for TRAIN and VAL splits."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.dataset import iter_build_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["TRAIN", "VAL"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    start = time.time()
    summary = iter_build_cache(tuple(args.splits), limit=args.limit)
    summary["elapsed_seconds"] = round(time.time() - start, 1)
    out = Path("results") / "cache_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
