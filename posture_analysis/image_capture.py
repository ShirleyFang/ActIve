import cv2
import os
import shutil
import time

class WebcamCapture:
    def __init__(self, save_dir="../data/img/user", num_images=1):
        """Initialize webcam capture settings."""
        self.save_dir = save_dir
        self.num_images = num_images
        os.makedirs(self.save_dir, exist_ok=True)

    def countdown(self, seconds):
        """Countdown before capturing a photo."""
        print(f"{seconds} seconds later, we will start capturing your photo...")

        for i in range(seconds, 0, -1):
            print(i)
            time.sleep(1)  # Uncomment to enable actual countdown timing

        print("📸 Capturing photo now!")

    def capture_images(self):
        """Captures multiple images from the webcam and saves them."""
        cap = cv2.VideoCapture(0)  # Open default webcam
        if not cap.isOpened():
            print("❌ Failed to open webcam.")
            return

        time.sleep(3)  # Small delay to allow camera to adjust
        self.countdown(5)

        for i in range(self.num_images):
            ret, frame = cap.read()
            if not ret:
                print(f"⚠ Failed to capture image {i+1}")
                continue

            img_path = os.path.join(self.save_dir, f"image_{i+1}.jpg")
            cv2.imwrite(img_path, frame)

            time.sleep(0.5)  # Small delay to capture distinct images

        cap.release()
        print("📸 All images captured!")

    def delete_folder(self):
        """Deletes the entire folder and its contents."""
        try:
            if os.path.exists(self.save_dir):
                shutil.rmtree(self.save_dir)
                print(f"🗑 Folder '{self.save_dir}' deleted successfully!")
            else:
                print(f"⚠ Warning: Folder '{self.save_dir}' not found.")
        except Exception as e:
            print(f"❌ Error deleting folder: {e}")
