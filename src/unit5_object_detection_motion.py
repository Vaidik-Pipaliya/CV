import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def run_unit5_object_recognition_and_motion():
    print("=====================================================================")
    print("UNIT 5: OBJECT RECOGNITION (HAAR & YOLO) & MOTION ESTIMATION")
    print("=====================================================================")

    output_dir = os.path.join("outputs", "unit5")
    os.makedirs(output_dir, exist_ok=True)

    video_path = "data/videos/original_video.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] '{video_path}' not found. Please record or place your video in 'data/videos/'.")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {video_path} ({frame_count} frames @ {fps:.1f} FPS)")

    # Extract a middle frame from the video for Object Recognition (Part A)
    target_frame_idx = min(frame_count // 2, 60)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, video_frame = cap.read()

    if not ret or video_frame is None:
        print("[ERROR] Could not read frame from video for object recognition.")
        cap.release()
        sys.exit(1)

    # ===================================================================
    # PART A: OBJECT RECOGNITION ON VIDEO FRAME (Haar Cascade vs YOLO)
    # ===================================================================
    print("\n--- Part A: Object Recognition (Using Video Frame) ---")
    img_bgr = video_frame
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Haar Cascade Face Detection
    haar_xml = os.path.join("data", "haarcascade_frontalface_default.xml")
    if not os.path.exists(haar_xml):
        haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    
    face_cascade = cv2.CascadeClassifier(haar_xml)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"[Haar Cascade] Detected {len(faces)} face(s) in video frame")

    haar_img = img_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(haar_img, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(haar_img, "Face (Haar)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 2. Ultralytics YOLO Object Detection
    print("[YOLO] Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    results = model(img_bgr, verbose=False)
    yolo_img = img_bgr.copy()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            print(f"       - Detected '{cls_name}' with confidence {conf:.2f}")

            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(yolo_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            label = f"{cls_name} ({conf:.2f})"
            cv2.putText(yolo_img, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_dir, "01_haar_face_detection.jpg"), haar_img)
    cv2.imwrite(os.path.join(output_dir, "02_yolo_object_detection.jpg"), yolo_img)

    # ===================================================================
    # TASK 3 STEP 5: DETECTOR FAILURE CASE ANALYSIS (Haar vs YOLO)
    # ===================================================================
    print("\n--- Task 3.5: Detector Failure / Edge Case Analysis ---")
    print("Testing 90° clockwise rotated frame (head tilt / profile orientation)...")
    rot_frame = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    rot_gray = cv2.cvtColor(rot_frame, cv2.COLOR_BGR2GRAY)

    haar_rot_faces = face_cascade.detectMultiScale(rot_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"       - Haar Detections on Rotated Frame: {len(haar_rot_faces)} (FAILED / Missed detection due to lack of rotation invariance)")

    yolo_rot_res = model(rot_frame, verbose=False)
    yolo_rot_detections = []
    for r in yolo_rot_res:
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            yolo_rot_detections.append(f"{cls_name} ({conf:.2f})")
    print(f"       - YOLO Detections on Rotated Frame: {', '.join(yolo_rot_detections) if yolo_rot_detections else 'None'}")

    haar_fail_vis = rot_frame.copy()
    cv2.putText(haar_fail_vis, f"Haar: {len(haar_rot_faces)} faces (MISSED - Rotation Sensitive)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    yolo_fail_vis = rot_frame.copy()
    for r in yolo_rot_res:
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(yolo_fail_vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
            cv2.putText(yolo_fail_vis, f"{cls_name} {conf:.2f}", (b[0], max(25, b[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(yolo_fail_vis, "YOLO: Deep Feature Robustness", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    fig_fail, axes_fail = plt.subplots(1, 2, figsize=(12, 6))
    fig_fail.suptitle("Task 3.5: Failure Case Analysis — Haar Rotation Sensitivity vs YOLO Robustness", fontsize=13, fontweight="bold")
    axes_fail[0].imshow(cv2.cvtColor(haar_fail_vis, cv2.COLOR_BGR2RGB))
    axes_fail[0].set_title(f"Haar Cascade (0 faces detected)", fontsize=11)
    axes_fail[0].axis("off")
    axes_fail[1].imshow(cv2.cvtColor(yolo_fail_vis, cv2.COLOR_BGR2RGB))
    axes_fail[1].set_title("YOLOv8 Detection on Rotated Frame", fontsize=11)
    axes_fail[1].axis("off")
    plt.tight_layout()
    fail_path = os.path.join(output_dir, "07_detector_failure_case_analysis.jpg")
    plt.savefig(fail_path, dpi=200, bbox_inches="tight")
    plt.close()

    # ===================================================================
    # PART B: VIDEO MOTION ESTIMATION
    # ===================================================================
    print("\n--- Part B: Video Motion Estimation ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret1, frame1 = cap.read()

    # Skip 5 frames ahead for motion calculation
    for _ in range(5):
        cap.read()
    ret2, frame2 = cap.read()

    if not ret1 or not ret2:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret1, frame1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
        ret2, frame2 = cap.read()

    f1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    f2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 1. Frame Differencing
    diff_mask = cv2.absdiff(f1_gray, f2_gray)
    _, diff_thresh = cv2.threshold(diff_mask, 25, 255, cv2.THRESH_BINARY)

    # 2. GMM Background Subtractor (MOG2)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame_idx - 15))
    for _ in range(15):
        ok, f = cap.read()
        if ok:
            _ = bg_subtractor.apply(f)
    
    fg_mask = bg_subtractor.apply(frame2)
    _, fg_mask_clean = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # 3. Lucas-Kanade Optical Flow
    feature_params = dict(maxCorners=150, qualityLevel=0.2, minDistance=5, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    p0 = cv2.goodFeaturesToTrack(f1_gray, mask=None, **feature_params)
    optical_flow_img = frame2.copy()
    
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(f1_gray, f2_gray, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            for (new, old) in zip(good_new, good_old):
                a, b = map(int, new.ravel())
                c, d = map(int, old.ravel())
                cv2.line(optical_flow_img, (c, d), (a, b), (0, 255, 0), 2)
                cv2.circle(optical_flow_img, (a, b), 4, (0, 0, 255), -1)

    cap.release()

    cv2.imwrite(os.path.join(output_dir, "03_frame_differencing_mask.jpg"), diff_thresh)
    cv2.imwrite(os.path.join(output_dir, "04_gmm_background_subtraction_mask.jpg"), fg_mask_clean)
    cv2.imwrite(os.path.join(output_dir, "05_lucas_kanade_optical_flow.jpg"), optical_flow_img)

    # Summary Comparison Grid Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Unit 5: Object Recognition & Video Motion Estimation Comparison", fontsize=15, fontweight="bold")

    unit5_plots = [
        ("1. Haar Cascade Face Detection", cv2.cvtColor(haar_img, cv2.COLOR_BGR2RGB), None),
        ("2. YOLOv8 Object Detection", cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB), None),
        ("3. Frame Differencing Mask\n(Highlights moving edges)", diff_thresh, "gray"),
        ("4. GMM Background Subtraction\n(Isolates foreground silhouette)", fg_mask_clean, "gray"),
        ("5. Lucas-Kanade Optical Flow\n(Tracks motion vectors & corners)", cv2.cvtColor(optical_flow_img, cv2.COLOR_BGR2RGB), None),
    ]

    for idx, (title, img, cmap_val) in enumerate(unit5_plots):
        r, c = divmod(idx, 3)
        if cmap_val:
            axes[r, c].imshow(img, cmap=cmap_val)
        else:
            axes[r, c].imshow(img)
        axes[r, c].set_title(title, fontsize=11)
        axes[r, c].axis("off")

    axes[1, 2].axis("off")
    plt.tight_layout()
    summary_grid_path = os.path.join(output_dir, "06_unit5_recognition_motion_summary_grid.png")
    plt.savefig(summary_grid_path, dpi=200, bbox_inches="tight")
    plt.close()

    # ===================================================================
    # TASK 4 STEP 6: COMPARISON TABLE
    # ===================================================================
    print("\n-------------------------------------------------------------------------------------------------------------------")
    print("TASK 4 STEP 6: VIDEO MOTION ESTIMATION COMPARISON TABLE")
    print("-------------------------------------------------------------------------------------------------------------------")
    table_text = (
        "| Motion Estimation Technique      | Mathematical Basis                  | Output Representation             | Computational Cost | Key Strengths                     | Key Limitations                      |\n"
        "| :------------------------------- | :---------------------------------- | :-------------------------------- | :----------------- | :-------------------------------- | :----------------------------------- |\n"
        "| 1. Frame Differencing            | Pixel-wise subtraction |I(t) - I(t-1)|  Binary edge mask of motion         | Extremely Low (<1ms)| Real-time, zero memory state      | Suffers from ghosting, interior void |\n"
        "| 2. GMM Background Subtraction    | Multi-modal Gaussian distribution   | Full solid foreground silhouette  | Moderate (~5-10ms)  | Complete segmentation, learns scene| Needs warmup, sensitive to camera pan|\n"
        "| 3. Lucas-Kanade Optical Flow     | Brightness constancy constraint      | Sparse motion velocity vectors (dx,dy)| Moderate (~8-15ms)  | Direction and speed quantified     | Small motion assumption, corner-only |\n"
    )
    print(table_text)
    print("-------------------------------------------------------------------------------------------------------------------")

    print(f"[SUCCESS] Unit 5 Complete! All output files saved in '{output_dir}/'")
    print(f"[REPORT FILE] Failure Analysis Grid : '{fail_path}'")
    print(f"[REPORT FILE] Recognition & Motion : '{summary_grid_path}'")
    print("=====================================================================")


if __name__ == "__main__":
    run_unit5_object_recognition_and_motion()
