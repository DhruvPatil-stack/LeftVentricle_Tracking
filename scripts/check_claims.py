"""Reject unsupported clinical claims and hand-entered percentages."""
import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
metrics = (root / "results" / "metrics.json").read_text(encoding="utf-8")
errors = []
for path in [root / "README.md", root / "docs" / "THEORY.md", root / "left_ventricle_tracking_project_write_up.md"]:
    text = path.read_text(encoding="utf-8").lower()
    for banned in ["clinically validated", "successfully isolates", "robust", "state of the art", "state-of-the-art"]:
        if banned in text: errors.append(f"{path.name}: banned claim {banned}")
    outside = re.sub(r"<!-- GENERATED METRICS START -->.*?<!-- GENERATED METRICS END -->", "", text, flags=re.S)
    for number in re.findall(r"\b\d+(?:\.\d+)?\s*percent\b", outside):
        if number not in metrics.lower(): errors.append(f"{path.name}: unsupported percentage {number}")
if errors:
    print("\n".join(errors)); sys.exit(1)
print("Claims check passed.")
