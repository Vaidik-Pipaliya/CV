import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def load_original_image(image_dir="data/images"):
    """
    Safely loads data/images/original.jpg without modifying the original file.
    Also checks common Windows naming mistakes (like original.jpg.jpg or .png).
    """
    expected_path = os.path.join(image_dir, "original.jpg")

    # Check if exact path exists, or look for alternative extensions
    if not os.path.exists(expected_path):
        alternatives = ["original.jpg.jpg", "original.jpeg", "original.png", "original.JPG"]
        for alt in alternatives:
            alt_path = os.path.join(image_dir, alt)
            if os.path.exists(alt_path):
                print(f"[INFO] Found '{alt}' instead of 'original.jpg'. Loading '{alt_path}'...")
                expected_path = alt_path
                break

    if not os.path.exists(expected_path):
        print(f"\n[ERROR] Could not find '{expected_path}'!")
        print("Please copy your photograph into 'data/images/' and name it 'original.jpg'.")
        sys.exit(1)

    img_bgr = cv2.imread(expected_path)
    if img_bgr is None:
        print(f"\n[ERROR] File at '{expected_path}' could not be read as an image.")
        sys.exit(1)

    # Convert from OpenCV BGR format to standard RGB format for visualization & metrics
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb, expected_path


def add_gaussian_noise(image_rgb, mean=0, sigma=25):
    """
    Task 1: Adds Gaussian (normal bell-curve) noise to an RGB image.
    sigma controls the intensity/spread of the grain.
    """
    rng = np.random.default_rng(seed=42)
    gaussian_noise = rng.normal(mean, sigma, image_rgb.shape)

    noisy_float = image_rgb.astype(np.float32) + gaussian_noise
    noisy_uint8 = np.clip(noisy_float, 0, 255).astype(np.uint8)
    return noisy_uint8


def add_salt_and_pepper_noise(image_rgb, salt_prob=0.02, pepper_prob=0.02):
    """
    Task 2: Adds Salt (pure white = 255) and Pepper (pure black = 0) impulse noise.
    2% of pixels become pure white, and 2% become pure black.
    """
    rng = np.random.default_rng(seed=42)
    noisy_image = image_rgb.copy()

    height, width, _ = image_rgb.shape
    random_matrix = rng.random((height, width))

    noisy_image[random_matrix < salt_prob] = [255, 255, 255]
    noisy_image[random_matrix > (1.0 - pepper_prob)] = [0, 0, 0]
    return noisy_image


def evaluate_quality(clean_rgb, test_rgb):
    """
    Task 5: Calculates PSNR (in dB) and SSIM (from 0 to 1) against the clean image.
    """
    psnr_val = compute_psnr(clean_rgb, test_rgb, data_range=255)
    ssim_val = compute_ssim(clean_rgb, test_rgb, data_range=255, channel_axis=2)
    return psnr_val, ssim_val


def run_unit3_partA_restoration():
    print("=====================================================================")
    print("UNIT 3 (PART A): IMAGE RESTORATION (NOISE, FILTERING, PSNR & SSIM)")
    print("=====================================================================")

    output_dir = os.path.join("outputs", "unit3")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the original clean photograph
    _, clean_rgb, loaded_path = load_original_image()
    h, w, c = clean_rgb.shape
    print(f"Loaded Original Image : {loaded_path} ({w}x{h} pixels)")

    # 2. Add Gaussian Noise and Salt-and-Pepper Noise
    print("Generating Gaussian noise and Salt-and-Pepper noise...")
    gaussian_noisy = add_gaussian_noise(clean_rgb, mean=0, sigma=25)
    sp_noisy = add_salt_and_pepper_noise(clean_rgb, salt_prob=0.02, pepper_prob=0.02)

    # 3. Apply Gaussian Filtering and Median Filtering to BOTH noisy images
    kernel_size = 5
    print(f"Applying {kernel_size}x{kernel_size} Gaussian Filter and Median Filter...")

    gauss_restored_with_gaussian = cv2.GaussianBlur(gaussian_noisy, (kernel_size, kernel_size), 0)
    gauss_restored_with_median = cv2.medianBlur(gaussian_noisy, kernel_size)

    sp_restored_with_gaussian = cv2.GaussianBlur(sp_noisy, (kernel_size, kernel_size), 0)
    sp_restored_with_median = cv2.medianBlur(sp_noisy, kernel_size)

    # 4. Calculate PSNR and SSIM for all images
    print("Calculating PSNR and SSIM metrics...")
    experiments = [
        ("Gaussian Noisy (Unfiltered)", gaussian_noisy, "01_gaussian_noisy.jpg"),
        ("Gaussian Noise -> Gaussian Filter", gauss_restored_with_gaussian, "02_gauss_noise_gaussian_filter.jpg"),
        ("Gaussian Noise -> Median Filter", gauss_restored_with_median, "03_gauss_noise_median_filter.jpg"),
        ("Salt & Pepper Noisy (Unfiltered)", sp_noisy, "04_sp_noisy.jpg"),
        ("S&P Noise -> Gaussian Filter", sp_restored_with_gaussian, "05_sp_noise_gaussian_filter.jpg"),
        ("S&P Noise -> Median Filter", sp_restored_with_median, "06_sp_noise_median_filter.jpg"),
    ]

    metrics_results = {}
    print("\n---------------------------------------------------------------------")
    print(f"{'Image / Restoration Method':<36} | {'PSNR (dB)':<12} | {'SSIM (0-1)':<10}")
    print("---------------------------------------------------------------------")

    for title, img_rgb, filename in experiments:
        psnr_val, ssim_val = evaluate_quality(clean_rgb, img_rgb)
        metrics_results[title] = (psnr_val, ssim_val)
        print(f"{title:<36} | {psnr_val:>9.2f} dB | {ssim_val:>9.4f}")

        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    print("---------------------------------------------------------------------")

    # 5. Create Side-by-Side Comparison Plot
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Unit 3A: Image Restoration Comparison (Gaussian vs. Median Filtering)", fontsize=15, fontweight="bold")

    row1_items = [
        ("Original Clean Image\n(Reference)", clean_rgb),
        (f"Gaussian Noisy\nPSNR: {metrics_results['Gaussian Noisy (Unfiltered)'][0]:.2f} dB | SSIM: {metrics_results['Gaussian Noisy (Unfiltered)'][1]:.3f}", gaussian_noisy),
        (f"Gaussian Noise + Gaussian Filter\nPSNR: {metrics_results['Gaussian Noise -> Gaussian Filter'][0]:.2f} dB | SSIM: {metrics_results['Gaussian Noise -> Gaussian Filter'][1]:.3f}", gauss_restored_with_gaussian),
        (f"Gaussian Noise + Median Filter\nPSNR: {metrics_results['Gaussian Noise -> Median Filter'][0]:.2f} dB | SSIM: {metrics_results['Gaussian Noise -> Median Filter'][1]:.3f}", gauss_restored_with_median),
    ]

    row2_items = [
        ("Original Clean Image\n(Reference)", clean_rgb),
        (f"Salt & Pepper Noisy\nPSNR: {metrics_results['Salt & Pepper Noisy (Unfiltered)'][0]:.2f} dB | SSIM: {metrics_results['Salt & Pepper Noisy (Unfiltered)'][1]:.3f}", sp_noisy),
        (f"S&P Noise + Gaussian Filter\nPSNR: {metrics_results['S&P Noise -> Gaussian Filter'][0]:.2f} dB | SSIM: {metrics_results['S&P Noise -> Gaussian Filter'][1]:.3f}", sp_restored_with_gaussian),
        (f"S&P Noise + Median Filter\nPSNR: {metrics_results['S&P Noise -> Median Filter'][0]:.2f} dB | SSIM: {metrics_results['S&P Noise -> Median Filter'][1]:.3f}", sp_restored_with_median),
    ]

    for col_idx, (title, img) in enumerate(row1_items):
        axes[0, col_idx].imshow(img)
        axes[0, col_idx].set_title(title, fontsize=10)
        axes[0, col_idx].axis("off")

    for col_idx, (title, img) in enumerate(row2_items):
        axes[1, col_idx].imshow(img)
        axes[1, col_idx].set_title(title, fontsize=10)
        axes[1, col_idx].axis("off")

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "07_restoration_comparison_grid.png")
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Saved 6 individual images + comparison grid to: '{output_dir}/'")
    print(f"[REPORT FILE] Main visual comparison saved at: '{comparison_path}'")


def run_unit3_partB_morphology_and_segmentation():
    print("\n=====================================================================")
    print("UNIT 3 (PART B): MORPHOLOGICAL PROCESSING & SEGMENTATION")
    print("=====================================================================")

    output_dir = os.path.join("outputs", "unit3")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load original clean image
    img_bgr, clean_rgb, _ = load_original_image()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Task 5: Thresholding (Otsu's Binarization)
    print("[Task 5] Applying binary thresholding (Otsu's binarization)...")
    otsu_thresh_val, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"         - Computed optimal Otsu threshold value: {otsu_thresh_val:.1f}")
    cv2.imwrite(os.path.join(output_dir, "08_thresholding_binary.jpg"), binary_otsu)

    # Task 6: Morphological Operations (Erosion, Dilation, Opening, Closing)
    print("[Task 6] Applying morphological operations (kernel 5x5)...")
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(binary_otsu, kernel, iterations=1)
    dilated = cv2.dilate(binary_otsu, kernel, iterations=1)
    opened = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel, iterations=1)

    cv2.imwrite(os.path.join(output_dir, "09_morphological_erosion.jpg"), eroded)
    cv2.imwrite(os.path.join(output_dir, "10_morphological_dilation.jpg"), dilated)
    cv2.imwrite(os.path.join(output_dir, "11_morphological_opening.jpg"), opened)
    cv2.imwrite(os.path.join(output_dir, "12_morphological_closing.jpg"), closed)

    # Task 7: Canny Edge Detection
    print("[Task 7] Applying Canny edge detection (thresholds 100, 200)...")
    canny_edges = cv2.Canny(gray, 100, 200)
    cv2.imwrite(os.path.join(output_dir, "13_canny_edge_detection.jpg"), canny_edges)

    # Task 8: Watershed Segmentation on touching objects
    print("[Task 8] Performing Watershed segmentation on touching objects...")
    touching_path = os.path.join("data", "images", "touching_objects.jpg")
    if not os.path.exists(touching_path):
        print(f"[WARNING] '{touching_path}' not found. Generating touching objects image...")
        from prepare_dataset import create_touching_objects_image
        create_touching_objects_image(touching_path)

    touch_bgr = cv2.imread(touching_path)
    touch_gray = cv2.cvtColor(touch_bgr, cv2.COLOR_BGR2GRAY)

    # Robust binarization for touching objects against dark background (~45 intensity)
    # Background slate is dark (~45), so threshold 55 cleanly isolates all objects (coins, caps, wood)
    _, touch_thresh = cv2.threshold(touch_gray, 55, 255, cv2.THRESH_BINARY)
    touch_kernel = np.ones((5, 5), np.uint8)
    touch_open = cv2.morphologyEx(touch_thresh, cv2.MORPH_OPEN, touch_kernel, iterations=2)
    # Morphological closing fills internal markings (engravings, bottle cap text, wood grain)
    touch_closed = cv2.morphologyEx(touch_open, cv2.MORPH_CLOSE, touch_kernel, iterations=3)
    sure_bg = cv2.dilate(touch_closed, touch_kernel, iterations=3)

    dist_transform = cv2.distanceTransform(touch_closed, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.55 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Filter out spurious microscopic seeds and record centroids
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    filtered_fg = np.zeros_like(sure_fg)
    valid_centroids = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 500:
            filtered_fg[labels == i] = 255
            valid_centroids.append((int(centroids[i][0]), int(centroids[i][1])))

    unknown = cv2.subtract(sure_bg, filtered_fg)

    # Marker labelling
    num_markers, markers = cv2.connectedComponents(filtered_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply Watershed
    watershed_img = touch_bgr.copy()
    markers = cv2.watershed(watershed_img, markers)

    # Color palette for distinct segment highlighting (BGR)
    palette = [
        [0, 255, 255],   # Yellow
        [0, 255, 0],     # Lime Green
        [255, 160, 0],   # Sky Blue
        [255, 0, 255],   # Magenta
        [0, 140, 255],   # Orange
        [200, 100, 255], # Purple / Pink
        [255, 255, 0],   # Cyan
        [0, 215, 255]    # Gold
    ]

    color_mask = np.zeros_like(touch_bgr, dtype=np.uint8)
    for idx, m_id in enumerate(range(2, num_markers + 1)):
        color_mask[markers == m_id] = palette[idx % len(palette)]

    # Make separating boundaries prominent and thick (7x7 dilation)
    boundary_mask = (markers == -1).astype(np.uint8)
    boundary_thick = cv2.dilate(boundary_mask, np.ones((7, 7), np.uint8), iterations=1)

    # Blend 65% original photo + 35% distinct segment color tint
    blended = cv2.addWeighted(touch_bgr, 0.65, color_mask, 0.35, 0)
    # Draw thick red boundary lines
    blended[boundary_thick == 1] = [0, 0, 255]

    # Draw object index numbers (#1, #2, etc.) at centroids
    for idx, (cx, cy) in enumerate(valid_centroids):
        cv2.putText(blended, f"#{idx+1}", (cx - 25, cy + 15), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(blended, f"#{idx+1}", (cx - 25, cy + 15), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)

    watershed_img = blended
    watershed_rgb = cv2.cvtColor(watershed_img, cv2.COLOR_BGR2RGB)

    num_separated = num_markers - 1
    print(f"         - Successfully identified and separated {num_separated} touching objects via Watershed!")
    cv2.imwrite(os.path.join(output_dir, "14_watershed_segmentation.jpg"), watershed_img)

    # Summary Grid for Part B (Morphology & Segmentation)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Unit 3B: Morphological Processing & Segmentation Results", fontsize=15, fontweight="bold")

    partB_items = [
        ("1. Input Grayscale", gray, "gray"),
        (f"2. Otsu Binary Mask (T={otsu_thresh_val:.0f})", binary_otsu, "gray"),
        ("3. Morphological Erosion (5x5)", eroded, "gray"),
        ("4. Morphological Dilation (5x5)", dilated, "gray"),
        ("5. Morphological Opening (Noise Removal)", opened, "gray"),
        ("6. Morphological Closing (Hole Filling)", closed, "gray"),
        ("7. Canny Edge Map (T1=100, T2=200)", canny_edges, "gray"),
        (f"8. Watershed Segmentation ({num_separated} Objects Separated)", watershed_rgb, None),
    ]

    for idx, (title, img, cmap_val) in enumerate(partB_items):
        r, c = divmod(idx, 4)
        if cmap_val:
            axes[r, c].imshow(img, cmap=cmap_val)
        else:
            axes[r, c].imshow(img)
        axes[r, c].set_title(title, fontsize=10)
        axes[r, c].axis("off")

    plt.tight_layout()
    grid_path = os.path.join(output_dir, "15_partB_morphology_segmentation_grid.png")
    plt.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Saved Part B individual images + summary grid to: '{output_dir}/'")
    print(f"[REPORT FILE] Part B Visual Summary saved at: '{grid_path}'")


if __name__ == "__main__":
    run_unit3_partA_restoration()
    run_unit3_partB_morphology_and_segmentation()
