import os
import sys
import cv2
import numpy as np


def load_user_portrait(image_dir="data/images"):
    """
    Finds the user's uploaded image inside data/images/, handles PNG transparency
    cleanly if present, and saves a standardized 'data/images/original.jpg'.
    """
    os.makedirs(image_dir, exist_ok=True)
    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    ignore_files = {"original.jpg", "second_view.jpg", "touching_objects.jpg"}

    candidates = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(valid_exts) and f.lower() not in ignore_files
    ]

    if candidates:
        source_path = os.path.join(image_dir, candidates[0])
    elif os.path.exists(os.path.join(image_dir, "original.jpg")):
        source_path = os.path.join(image_dir, "original.jpg")
    else:
        print("=================================================================")
        print("[ERROR] No image found inside 'data/images/'!")
        print("1. Run in terminal:  explorer data\\images")
        print("2. Paste your image into that folder.")
        print("3. Run again:        python src/prepare_dataset.py")
        print("=================================================================")
        sys.exit(1)

    raw = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"[ERROR] Could not read '{source_path}' as an image.")
        sys.exit(1)

    if len(raw.shape) == 3 and raw.shape[2] == 4:
        bgr = raw[:, :, :3]
        alpha = raw[:, :, 3].astype(np.float32) / 255.0
        alpha_3c = np.dstack([alpha, alpha, alpha])
        white_bg = np.full_like(bgr, 255, dtype=np.uint8)
        clean_bgr = (bgr * alpha_3c + white_bg * (1.0 - alpha_3c)).astype(np.uint8)
        person_mask = (raw[:, :, 3] > 30).astype(np.uint8) * 255
    else:
        clean_bgr = raw if len(raw.shape) == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
        person_mask = (gray < 245).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)

    target_original = os.path.join(image_dir, "original.jpg")
    cv2.imwrite(target_original, clean_bgr)
    h, w = clean_bgr.shape[:2]
    print(f"[OK] Saved main photograph         : '{target_original}' ({w}x{h} pixels)")
    return clean_bgr, person_mask


def create_second_view(original_bgr, save_path="data/images/second_view.jpg"):
    h, w = original_bgr.shape[:2]
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dx = int(w * 0.06)
    dy = int(h * 0.05)
    dst_pts = np.float32([
        [dx, dy],
        [w - 1 - int(dx * 0.4), int(dy * 0.6)],
        [w - 1 - dx, h - 1 - dy],
        [int(dx * 0.3), h - 1 - int(dy * 0.4)]
    ])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        original_bgr, matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    second_view = np.clip(warped.astype(np.float32) * 0.97 + 4, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, second_view)
    print(f"[OK] Generated second view image   : '{save_path}' ({w}x{h} pixels)")


def create_touching_objects_image(save_path="data/images/touching_objects.jpg"):
    canvas_h, canvas_w = 500, 650
    img = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)
    coins = [
        (160, 180, 68),
        (275, 200, 68),
        (215, 300, 65),
        (430, 175, 68),
        (475, 285, 68),
        (375, 360, 64),
    ]
    for (cx, cy, r) in coins:
        cv2.circle(img, (cx, cy), r, (205, 215, 225), -1)
        cv2.circle(img, (cx, cy), int(r * 0.82), (235, 240, 245), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(save_path, img)
    print(f"[OK] Generated watershed image     : '{save_path}' ({canvas_w}x{canvas_h} pixels)")


if __name__ == "__main__":
    print("=== Building Assignment Dataset from Portrait ===")
    clean_img, _ = load_user_portrait("data/images")
    create_second_view(clean_img, "data/images/second_view.jpg")
    create_touching_objects_image("data/images/touching_objects.jpg")
    print("================================================")
    print("[SUCCESS] All dataset files are ready!")