import cv2
import os
import shutil
import time


def capture_images(save_dir="../data/img/user", num_images=1):
    """Captures multiple images from the webcam and saves them."""
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)  # Open default webcam
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    print("📸 Capturing images...")
    time.sleep(3)
    for i in range(num_images):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠ Failed to capture image {i+1}")
            continue

        img_path = os.path.join(save_dir, f"image_{i+1}.jpg")
        cv2.imwrite(img_path, frame)

        time.sleep(0.5)  # Small delay to capture distinct images


    cap.release()
    print("📸 All images captured!")


def delete_folder(folder_path):
    """Deletes the entire folder and its contents."""
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"🗑 Folder '{folder_path}' deleted successfully!")
        else:
            print(f"⚠ Warning: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"❌ Error deleting folder: {e}")
