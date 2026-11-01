import os
import urllib.request
import cv2
import numpy as np
import matplotlib
import skimage
import ultralytics

print("=== Phase 0: Environment & Dataset Verification ===")
print(f"OpenCV Version       : {cv2.__version__}")
print(f"NumPy Version        : {np.__version__}")
print(f"Matplotlib Version   : {matplotlib.__version__}")
print(f"scikit-image Version : {skimage.__version__}")
print(f"Ultralytics Version  : {ultralytics.__version__}")

# 1. Verify SIFT Support (Unit 4)
try:
    sift = cv2.SIFT_create()
    print("SIFT Support         : AVAILABLE (Ready for Unit 4)")
except Exception as e:
    print(f"SIFT Support         : ERROR ({e})")

# 2. Verify or Download Haar Cascade XML File (Unit 5)
# In OpenCV 5.0.0, XML files are not bundled in cv2.data, so we store it in data/
haar_local_path = os.path.join("data", "haarcascade_frontalface_default.xml")
haar_url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"

if not os.path.exists(haar_local_path):
    print("Downloading official Haar Cascade XML file to 'data/'...")
    try:
        urllib.request.urlretrieve(haar_url, haar_local_path)
    except Exception as e:
        print(f"Download Error       : {e}")

# Load the XML into CascadeClassifier to make sure it is valid
cascade = cv2.CascadeClassifier(haar_local_path)
if not cascade.empty():
    print(f"Haar Cascade File    : READY ({haar_local_path})")
else:
    print("Haar Cascade File    : ERROR (Could not load XML)")

# 3. Verify Dataset Files (Photos and Video)
print("\n=== Dataset Check ===")
image_files = [
    ("Main Photograph (original.jpg)", "data/images/original.jpg"),
    ("Second View (second_view.jpg)", "data/images/second_view.jpg"),
    ("Watershed Image (touching_objects.jpg - Optional)", "data/images/touching_objects.jpg")
]

for label, path in image_files:
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            h, w, c = img.shape
            print(f"[FOUND] {label}: {w}x{h} pixels")
        else:
            print(f"[ERROR] {label}: File exists at {path} but could not be opened as an image.")
    else:
        print(f"[MISSING] {label}: Please place file at '{path}'")

video_path = "data/videos/original_video.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[FOUND] Original Video (original_video.mp4): {w}x{h} pixels, {duration:.1f} seconds ({frame_count} frames)")
        cap.release()
    else:
        print(f"[ERROR] Video exists at {video_path} but could not be opened.")
else:
    print(f"[MISSING] Original Video: Please place a 15-30s video at '{video_path}'")

print("===================================================")