import numpy as np
import torch
import torch.nn as nn
import cv2


# ---------------------------------------------------------------------------
# Minimal U-Net implementation compatible with standard pre-trained weights
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: list[int] | tuple[int, ...] | None = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------

TARGET_SIZE = (112, 112)


def choose_device(requested: str | None = None) -> str:
    """Use CUDA when available unless the caller requests a device."""
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    weights_path: str,
    device: str | None = None,
    widths: list[int] | tuple[int, ...] | None = None,
) -> UNet:
    """Load matching U-Net weights and fail clearly on key mismatch."""
    device = choose_device(device)
    model = UNet(in_channels=1, out_channels=1, features=widths)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Weights must contain a U-Net state dictionary.")
    expected, received = set(model.state_dict()), set(state)
    if expected != received:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise ValueError(f"Weight keys do not match this U-Net. Missing: {missing}. Unexpected: {unexpected}.")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def frame_to_tensor(gray_frame: np.ndarray, target_size: tuple[int, int] = TARGET_SIZE) -> torch.Tensor:
    """Resize, normalise, and convert a grayscale frame to a (1,1,H,W) tensor."""
    resized = cv2.resize(gray_frame, target_size)
    normalised = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalised).unsqueeze(0).unsqueeze(0)
    return tensor


def predict_mask(
    model: UNet,
    gray_frame: np.ndarray,
    threshold: float = 0.5,
    device: str = "cpu",
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """Run inference on one preprocessed grayscale frame; return a binary uint8 mask."""
    tensor = frame_to_tensor(gray_frame, target_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary = (prob > threshold).astype(np.uint8) * 255
    original_size = (gray_frame.shape[1], gray_frame.shape[0])
    mask = cv2.resize(binary, original_size, interpolation=cv2.INTER_NEAREST)
    return mask


def segment_frames(
    model: UNet,
    preprocessed_frames: list[np.ndarray],
    threshold: float = 0.5,
    device: str | None = None,
    batch_size: int = 16,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> list[np.ndarray]:
    """Return binary masks with batched inference."""
    if not preprocessed_frames:
        return []
    device = choose_device(device)
    masks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(preprocessed_frames), batch_size):
        frames = preprocessed_frames[start : start + batch_size]
        tensors = torch.cat([frame_to_tensor(frame, target_size) for frame in frames]).to(device)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(tensors)).squeeze(1).cpu().numpy()
        for probability, frame in zip(probabilities, frames):
            binary = (probability > threshold).astype(np.uint8) * 255
            masks.append(cv2.resize(binary, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST))
    return masks
