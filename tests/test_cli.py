import json
import subprocess
import sys
import cv2
import numpy as np
import torch
from src.segment import UNet


def test_cli_smoke_with_synthetic_video(tmp_path):
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 50, (64, 64))
    for _ in range(5): writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    weights = tmp_path / "weights.pth"
    model = UNet()
    with torch.no_grad():
        model.final.weight.zero_()
        model.final.bias.fill_(10.0)
    torch.save(model.state_dict(), weights)
    output = tmp_path / "out"
    completed = subprocess.run([sys.executable, "-m", "src.run", "--video", str(video), "--weights", str(weights), "--out", str(output)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert (output / "result.json").exists()
    assert json.loads((output / "result.json").read_text())["metadata"]["frame_count"] == 5
