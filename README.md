# Automated Left Ventricle Segmentation and Ejection Fraction Estimation

## Overview

This project bridges 2D medical image processing with automated computer vision pipelines. Utilizing clinical echocardiogram video data, this repository implements a tracking system to isolate the left ventricle (LV) across the cardiac cycle. The pipeline acts as the foundational step for automating critical cardiovascular metrics like Ejection Fraction (EF).

---

## Objective

Build a robust, automated pipeline capable of:
- Processing dynamic cardiovascular ultrasound (`.avi`) videos from the EchoNet-Dynamic dataset
- Handling inherent ultrasound speckle noise without sacrificing edge fidelity
- Extracting clinically relevant spatial measurements frame-by-frame
- Estimating Ejection Fraction from pixel-level LV area tracking

---

## Methodology

### 1. Data Pipeline
Ingests raw apical 4-chamber (A4C) view echocardiogram `.avi` files from Stanford's open-source **EchoNet-Dynamic** dataset and extracts them into sequential NumPy frame arrays via `cv2.VideoCapture`.

### 2. Pre-processing & Noise Reduction
Applies a two-stage denoising strategy:
- **Median filter** (`cv2.medianBlur`) — removes salt-and-pepper / speckle noise
- **Bilateral filter** (`cv2.bilateralFilter`) — smooths homogeneous tissue regions while preserving sharp myocardium edge boundaries

### 3. Segmentation
Loads a **pre-trained U-Net** segmentation architecture (PyTorch) adapted for medical imaging. The model runs per-frame inference to produce binary masks isolating the left ventricular cavity from surrounding tissue.

### 4. Contour Tracking
Applies `cv2.findContours` (external contours, simple chain approximation) to each binary mask. The largest contour per frame is selected as the LV wall boundary, mathematically tracking its shape through systole and diastole.

### 5. Ejection Fraction Estimation
Calculates the 2D pixel area enclosed by the LV contour for every frame. The pipeline identifies:
- **End-Diastolic (ED) frame** — frame with maximum LV area (ventricle fully filled)
- **End-Systolic (ES) frame** — frame with minimum LV area (ventricle fully contracted)

EF is estimated as:

```
EF (%) = (ED_area - ES_area) / ED_area × 100
```

---

## Project Structure

```
LeftVentricle_Tracking/
├── data/               # Place EchoNet-Dynamic .avi files here (gitignored)
├── models/             # Place pre-trained .pth weights here (gitignored)
├── outputs/            # Generated plots and CSVs saved here
├── src/
│   ├── extract_frames.py       # .avi → sequential frame arrays
│   ├── preprocess.py           # Median + bilateral noise reduction
│   ├── segment.py              # U-Net inference → binary masks
│   ├── contour_tracking.py     # cv2.findContours → LV boundary tracking
│   └── ef_calculation.py       # Area curve → ED/ES detection → EF %
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| OpenCV (`cv2`) | Video I/O, filtering, contour detection |
| PyTorch | U-Net model loading and inference |
| NumPy | Frame array manipulation |
| Matplotlib | LV area curve visualisation |

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/DhruvPatil-stack/LeftVentricle_Tracking.git
cd LeftVentricle_Tracking

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add data
#    Place EchoNet-Dynamic .avi files in data/
#    Place pre-trained U-Net weights (.pth) in models/
```

---

## Usage

```python
from src.extract_frames import extract_frames
from src.preprocess import preprocess_frames
from src.segment import load_model, segment_frames
from src.contour_tracking import track_contours
from src.ef_calculation import compute_lv_areas, estimate_ejection_fraction, plot_area_curve

# Load and preprocess
frames = extract_frames("data/sample.avi")
preprocessed = preprocess_frames(frames)

# Segment
model = load_model("models/unet_lv.pth")
masks = segment_frames(model, preprocessed)

# Track and measure
contours = track_contours(masks)
areas = compute_lv_areas(contours)

# Estimate EF
result = estimate_ejection_fraction(areas)
print(f"Ejection Fraction: {result['ejection_fraction_pct']}%")
print(f"ED Frame: {result['ed_frame_index']}  |  ES Frame: {result['es_frame_index']}")

# Visualise
plot_area_curve(areas, result['ed_frame_index'], result['es_frame_index'],
                save_path="outputs/lv_area_curve.png")
```

---

## Results & Clinical Impact

This pipeline successfully isolates and tracks the left ventricle in noisy, real-world clinical echocardiogram data. Key contributions:

- **End-to-end automation** from raw `.avi` video to EF percentage with no manual labelling required at inference time
- **Clinically grounded metrics** — ED and ES frame identification directly maps to the standard Simpson's method used in echocardiographic EF assessment
- **Practical robustness** — two-stage denoising preserves the myocardium boundary in low signal-to-noise ultrasound conditions
- **Modular design** — each processing stage is independently testable and swappable (e.g., replacing U-Net with SAM or nnU-Net)

This demonstrates practical skills in 2D medical image processing, deep learning model deployment, and cardiovascular-specific AI applications relevant to clinical decision support systems.

---

## Dataset

**EchoNet-Dynamic** — Stanford University  
Ouyang et al., *"Video-based AI for beat-to-beat assessment of cardiac function"*, Nature 2020.  
Access: [echonet.github.io/dynamic](https://echonet.github.io/dynamic/)
