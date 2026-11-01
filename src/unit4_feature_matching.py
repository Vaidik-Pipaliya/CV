import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure


def run_unit4_feature_extraction_and_matching():
    print("=====================================================================")
    print("UNIT 4: FEATURE EXTRACTION (HOG) & SIFT FEATURE MATCHING")
    print("=====================================================================")

    output_dir = os.path.join("outputs", "unit4")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Original Image and Second View Image
    orig_path = "data/images/original.jpg"
    second_path = "data/images/second_view.jpg"

    if not os.path.exists(orig_path) or not os.path.exists(second_path):
        print(f"[ERROR] Missing images in 'data/images/'. Please run 'python src/prepare_dataset.py' first.")
        sys.exit(1)

    img1_bgr = cv2.imread(orig_path)
    img2_bgr = cv2.imread(second_path)

    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_GRAY2BGR if len(img1_bgr.shape)==2 else cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_GRAY2BGR if len(img2_bgr.shape)==2 else cv2.COLOR_BGR2GRAY)

    # ===================================================================
    # TASK 1: HOG (Histogram of Oriented Gradients) Feature Extraction
    # ===================================================================
    print("[Task 1] Extracting HOG features from photograph...")
    # Resize slightly for clean HOG visualization
    hog_sz = (256, 256)
    hog_input = cv2.resize(gray1, hog_sz)

    hog_features, hog_image = hog(
        hog_input,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=None
    )
    # Rescale HOG image intensity for clear plotting
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    print(f"         - Extracted {len(hog_features)} HOG feature descriptors")

    # Save HOG visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.resize(img1_rgb, hog_sz))
    axes[0].set_title("Resized Input Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(hog_image_rescaled, cmap="gray")
    axes[1].set_title("HOG Gradient Orientations", fontsize=12)
    axes[1].axis("off")
    plt.tight_layout()
    hog_path = os.path.join(output_dir, "01_hog_feature_visualization.png")
    plt.savefig(hog_path, dpi=200, bbox_inches="tight")
    plt.close()

    # ===================================================================
    # TASKS 2, 3 & 4: SIFT Keypoints, Descriptors & Matching
    # ===================================================================
    print("[Task 2 & 3] Detecting SIFT keypoints and calculating descriptors...")
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"         - Image 1 SIFT Keypoints : {len(kp1)}")
    print(f"         - Image 2 SIFT Keypoints : {len(kp2)}")

    # Draw SIFT keypoints on original image
    img1_sift_kp = cv2.drawKeypoints(
        img1_rgb, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(
        os.path.join(output_dir, "02_sift_keypoints_image1.jpg"),
        cv2.cvtColor(img1_sift_kp, cv2.COLOR_RGB2BGR)
    )

    # [Task 4] Match SIFT features between Image 1 and Image 2 (Original vs Second View)
    print("[Task 4] Matching SIFT features between original and second view...")
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_view = bf.match(des1, des2)
    # Sort matches by distance (best matches first)
    matches_view = sorted(matches_view, key=lambda x: x.distance)
    print(f"         - Total SIFT matches found : {len(matches_view)}")

    # Draw top 50 matches
    match_img_view = cv2.drawMatches(
        img1_rgb, kp1, img2_rgb, kp2, matches_view[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(
        os.path.join(output_dir, "03_sift_matching_original_vs_second_view.jpg"),
        cv2.cvtColor(match_img_view, cv2.COLOR_RGB2BGR)
    )

    # ===================================================================
    # TASK 5: Repeat Matching After Rotation and Resizing
    # ===================================================================
    print("[Task 5] Testing SIFT invariance under Rotation and Resizing...")
    h, w = gray1.shape[:2]

    # Experiment A: Rotated Image (Rotate 30 degrees clockwise)
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
    rotated_img = cv2.warpAffine(gray1, rot_matrix, (w, h), borderValue=(255, 255, 255))
    rotated_rgb = cv2.warpAffine(img1_rgb, rot_matrix, (w, h), borderValue=(255, 255, 255))

    kp_rot, des_rot = sift.detectAndCompute(rotated_img, None)
    matches_rot = bf.match(des1, des_rot)
    matches_rot = sorted(matches_rot, key=lambda x: x.distance)
    print(f"         - Rotation (30 deg) Matches: {len(matches_rot)}")

    match_img_rot = cv2.drawMatches(
        img1_rgb, kp1, rotated_rgb, kp_rot, matches_rot[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(
        os.path.join(output_dir, "04_sift_matching_rotated.jpg"),
        cv2.cvtColor(match_img_rot, cv2.COLOR_RGB2BGR)
    )

    # Experiment B: Resized Image (Scale down to 70%)
    resized_img = cv2.resize(gray1, (int(w * 0.7), int(h * 0.7)))
    resized_rgb = cv2.resize(img1_rgb, (int(w * 0.7), int(h * 0.7)))

    kp_res, des_res = sift.detectAndCompute(resized_img, None)
    matches_res = bf.match(des1, des_res)
    matches_res = sorted(matches_res, key=lambda x: x.distance)
    print(f"         - Resized (70%) Matches    : {len(matches_res)}")

    match_img_res = cv2.drawMatches(
        img1_rgb, kp1, resized_rgb, kp_res, matches_res[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(
        os.path.join(output_dir, "05_sift_matching_resized.jpg"),
        cv2.cvtColor(match_img_res, cv2.COLOR_RGB2BGR)
    )

    # ===================================================================
    # TASK 6: Summary Comparison Grid Plot
    # ===================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Unit 4: SIFT Feature Extraction & Invariance Matching Comparison", fontsize=15, fontweight="bold")

    axes[0, 0].imshow(cv2.drawKeypoints(img1_rgb, kp1[:100], None))
    axes[0, 0].set_title(f"1. SIFT Keypoints (Top 100 shown: {len(kp1)} total)", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(match_img_view)
    axes[0, 1].set_title(f"2. Original vs. Second View ({len(matches_view)} matches)", fontsize=11)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(match_img_rot)
    axes[1, 0].set_title(f"3. Rotation 30° Invariance ({len(matches_rot)} matches)", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(match_img_res)
    axes[1, 1].set_title(f"4. Resize 70% Invariance ({len(matches_res)} matches)", fontsize=11)
    axes[1, 1].axis("off")

    plt.tight_layout()
    summary_grid_path = os.path.join(output_dir, "06_unit4_sift_matching_summary_grid.png")
    plt.savefig(summary_grid_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("---------------------------------------------------------------------")
    print(f"[SUCCESS] Unit 4 Complete! All output files saved in '{output_dir}/'")
    print(f"[REPORT FILE] SIFT Summary Grid: '{summary_grid_path}'")
    print("=====================================================================")


if __name__ == "__main__":
    run_unit4_feature_extraction_and_matching()
    