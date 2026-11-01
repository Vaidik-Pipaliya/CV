# CCA 3: Computer Vision Practical Coding Assignment
**Course Units Covered:** Units 3, 4, and 5  
**Programming Language:** Python 3.10+  
**Core Libraries:** OpenCV (`cv2`), NumPy, Matplotlib, scikit-image, Ultralytics YOLO  
**Evaluator:** Prof. Rahul Inchal (`rahul.inchal.inchal24@gmail.com`, `rahul.inchal.vf@mitwpu.edu.in`)  

---

## 1. Project Overview

This repository contains the complete implementation and technical documentation for the **CCA 3 Computer Vision Practical Assignment**, covering:

1. **Unit 3: Image Restoration & Segmentation**
   - Gaussian Noise & Salt-and-Pepper Noise generation.
   - Gaussian Filtering vs. Median Filtering with **PSNR** & **SSIM** quality assessment.
   - Otsu Binarization & Adaptive Thresholding.
   - Morphological operations (Erosion, Dilation, Opening, Closing).
   - Multi-stage Canny Edge Detection.
   - Distance Transform & Watershed Segmentation on touching objects.
2. **Unit 4: Feature Extraction & Matching**
   - Histogram of Oriented Gradients (**HOG**) visualization.
   - Scale-Invariant Feature Transform (**SIFT**) keypoint and descriptor extraction.
   - Feature matching between multiple perspective views.
   - Transformation invariance testing under **rotation ($30^\circ$)** and **spatial resizing ($70\%$)**.
3. **Unit 5: Object Recognition & Motion Estimation**
   - Viola–Jones **Haar Cascade** face detection.
   - Deep Learning object detection using **Ultralytics YOLOv8**.
   - Detector Failure Case Analysis (rotation sensitivity).
   - Video motion estimation using **Frame Differencing**, **GMM Background Subtraction (MOG2)**, and **Lucas–Kanade Optical Flow**.

---

## 2. Directory Structure

```text
CCA3-Computer-Vision/
├── CCA 3 CV - Assignment.pdf         # Official Assignment Specification
├── requirements.txt                  # Python Package Dependencies
├── .gitignore                        # Git ignore rules for venv/caches
├── README.md                         # Project documentation and execution guide
├── data/
│   ├── haarcascade_frontalface_default.xml # Pretrained Haar Cascade model
│   ├── images/
│   │   ├── original.jpg              # Original clean photograph
│   │   ├── second_view.jpg           # Second perspective view photograph
│   │   └── touching_objects.jpg      # Multi-object image for Watershed segmentation
│   └── videos/
│       └── original_video.mp4        # Original 10–30s motion video
├── src/
│   ├── check_env.py                  # Environment & dataset sanity checker
│   ├── prepare_dataset.py            # Dataset preparation and synthetic view generator
│   ├── unit3_restoration_segmentation.py # Unit 3: Restoration, Morphology & Watershed
│   ├── unit4_feature_matching.py     # Unit 4: HOG & SIFT Invariance Matching
│   └── unit5_object_detection_motion.py  # Unit 5: Haar, YOLOv8 & Motion Estimation
├── outputs/
│   ├── unit3/                        # 15 restoration & segmentation output images + grids
│   ├── unit4/                        # 6 HOG & SIFT matching output images + grids
│   └── unit5/                        # 7 detection, failure analysis & motion masks/grids
└── report/
    └── TECHNICAL_REPORT.md           # Full technical report with formulas, tables & captions
```

---

## 3. Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/Vaidik-Pipaliya/CV.git
cd CV
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Environment & Assets
```bash
python src/check_env.py
```

---

## 4. Execution Guide (Step-by-Step)

You can run each unit independently using the dedicated scripts in `src/`:

### 1. Dataset Preparation (Optional if using existing dataset)
```bash
python src/prepare_dataset.py
```

### 2. Run Unit 3 (Image Restoration & Segmentation)
```bash
python src/unit3_restoration_segmentation.py
```
*Outputs generated in `outputs/unit3/`:*
- `01_gaussian_noisy.jpg` to `06_sp_noise_median_filter.jpg`
- `07_restoration_comparison_grid.png` (Side-by-side restoration grid with PSNR/SSIM)
- `08_thresholding_binary.jpg` (Otsu threshold mask)
- `09_morphological_erosion.jpg` to `12_morphological_closing.jpg`
- `13_canny_edge_detection.jpg` (Canny edge map)
- `14_watershed_segmentation.jpg` (6 separated touching objects)
- `15_partB_morphology_segmentation_grid.png` (Part B summary grid)

### 3. Run Unit 4 (Feature Extraction & SIFT Matching)
```bash
python src/unit4_feature_matching.py
```
*Outputs generated in `outputs/unit4/`:*
- `01_hog_feature_visualization.png` (HOG gradient orientations)
- `02_sift_keypoints_image1.jpg` (SIFT rich keypoints)
- `03_sift_matching_original_vs_second_view.jpg` (Matching against second view)
- `04_sift_matching_rotated.jpg` (Rotation invariance test at $30^\circ$)
- `05_sift_matching_resized.jpg` (Scale invariance test at $70\%$)
- `06_unit4_sift_matching_summary_grid.png` (SIFT 4-panel comparison grid)

### 4. Run Unit 5 (Object Recognition & Motion Estimation)
```bash
python src/unit5_object_detection_motion.py
```
*Outputs generated in `outputs/unit5/`:*
- `01_haar_face_detection.jpg` (Haar Cascade bounding boxes)
- `02_yolo_object_detection.jpg` (YOLOv8 bounding boxes with confidence scores)
- `03_frame_differencing_mask.jpg` (Temporal differencing mask)
- `04_gmm_background_subtraction_mask.jpg` (MOG2 Gaussian mixture foreground mask)
- `05_lucas_kanade_optical_flow.jpg` (Sparse motion displacement vectors)
- `06_unit5_recognition_motion_summary_grid.png` (Recognition & motion grid)
- `07_detector_failure_case_analysis.jpg` (Haar $90^\circ$ failure vs. YOLO robustness)

---

## 5. Experimental Results Summary

### Unit 3 Restoration Performance
* **Salt & Pepper Noise:** Median Filter achieved **35.97 dB PSNR** and **0.9531 SSIM** (vastly outperforming Gaussian filter at 28.32 dB / 0.6603 SSIM).
* **Gaussian Noise:** Median Filter achieved **30.54 dB PSNR** (vs. 29.95 dB for Gaussian filter) while preserving sharp boundary edges.

### Unit 4 SIFT Feature Invariance
* **Original vs. 2nd View:** 2,723 valid matches.
* **Rotation Invariance ($30^\circ$):** 2,876 valid matches.
* **Scale Invariance ($70\%$):** 1,850 valid matches.

### Unit 5 Motion Estimation Techniques
* **Frame Differencing:** Fastest (<1ms), ideal for high-speed motion triggering, but leaves hollow object interiors.
* **GMM Background Subtraction:** Produces solid silhouette masks, automatically adapts to illumination, best for static cameras.
* **Lucas–Kanade Optical Flow:** Quantifies both velocity and direction vectors across moving corners.

---

## 6. Submission Details & Links

* **Detailed Technical Documentation:** See [`report/TECHNICAL_REPORT.md`](report/TECHNICAL_REPORT.md)
* **Git Repository Access:** Grant collaborator access to:
  * `rahul.inchal.inchal24@gmail.com`
  * `rahul.inchal.vf@mitwpu.edu.in`
* **Video Presentation Link:** *[Insert Google Drive Link Here]*
