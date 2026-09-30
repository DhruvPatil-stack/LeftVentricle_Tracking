# Project: Automated Left Ventricle Segmentation and Ejection Fraction Estimation

## Overview
This project bridges 2D medical image processing with automated computer vision pipelines. Utilizing clinical echocardiogram video data, I developed a tracking system to isolate the left ventricle (LV) across the cardiac cycle. This pipeline acts as the foundational step for automating critical cardiovascular metrics like Ejection Fraction (EF).

## Objective
To build a robust, automated pipeline capable of processing dynamic cardiovascular ultrasound videos, handling inherent speckle noise, and extracting clinically relevant spatial measurements frame-by-frame.

## Methodology
1. **Data Pipeline:** Ingested raw apical 4-chamber (A4C) view echocardiogram `.avi` files from Stanford's open-source EchoNet-Dynamic dataset, extracting them into sequential frame arrays.
2. **Pre-processing & Noise Reduction:** Applied median and bilateral filtering (via OpenCV) to reduce standard ultrasound speckle noise while preserving critical edge boundaries of the heart muscle.
3. **Segmentation & Tracking:** 
   - Utilized a pre-trained U-Net segmentation architecture adapted for medical imaging to generate binary masks of the left ventricular cavity.
   - Applied contour detection (`cv2.findContours`) on the resulting masks to mathematically track the changing boundaries of the LV wall throughout systole and diastole.
4. **Metric Calculation:** Calculated the 2D pixel area of the isolated ventricle frame-by-frame to identify the end-diastolic (maximum area) and end-systolic (minimum area) frames, providing the baseline data needed to estimate the EF percentage.

## Tech Stack
* **Language:** Python
* **Libraries:** OpenCV (video/image processing), PyTorch (segmentation model deployment), NumPy, Matplotlib

## Results & Impact
Engineered an end-to-end processing pipeline that successfully isolates and tracks the left ventricle in noisy, real-world clinical data. This demonstrates strong practical skills in 2D image processing, computer vision integration, and cardiovascular-specific AI applications.