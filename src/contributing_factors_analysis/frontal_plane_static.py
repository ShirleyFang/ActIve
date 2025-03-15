import cv2
import mediapipe as mp
import numpy as np
import os

class FrontalPlanePostureAnalyzer:
    
    def __init__(self):
        """Initialize MediaPipe Pose Estimation."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True)
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)  # First point
        b = np.array(b)  # Middle joint
        c = np.array(c)  # Last joint

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def image_analysis(self, img_path):
        """Analyze posture from an image."""
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe
        result = self.pose.process(image_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Extract key points
            shoulder_left = [landmarks[11].x, landmarks[11].y]
            shoulder_right = [landmarks[12].x, landmarks[12].y]
            hip_left = [landmarks[23].x, landmarks[23].y]
            hip_right = [landmarks[24].x, landmarks[24].y]
            knee_left = [landmarks[25].x, landmarks[25].y]
            knee_right = [landmarks[26].x, landmarks[26].y]
            ankle_left = [landmarks[27].x, landmarks[27].y]
            ankle_right = [landmarks[28].x, landmarks[28].y]

            # 🟡 Shoulder Asymmetry Detection
            shoulder_diff = abs(shoulder_left[1] - shoulder_right[1])
            # 🟠 Hip Misalignment Detection
            hip_diff = abs(hip_left[1] - hip_right[1])
            # 🔴 Knee Valgus (Knock Knees) Detection
            knee_angle_left = self.calculate_angle(hip_left, knee_left, ankle_left)
            knee_angle_right = self.calculate_angle(hip_right, knee_right, ankle_right)

            return (shoulder_diff, hip_diff, knee_angle_left, knee_angle_right)
        else:
            print("No pose detected in the image.")
            return None

    def image_process(self):
        """Process multiple images from the user folder."""
        image_path = "user/"
        image_data_list = []

        for i in range(1):
            full_image_path = os.path.join(image_path, f"image_{i+1}.jpg")
            curr_result = self.image_analysis(full_image_path)
            if curr_result:
                image_data_list.append(curr_result)

        return image_data_list

    def average_image_data(self, image_data_list):
        """Compute the average values from multiple images."""
        shoulder_diff_total, hip_diff_total, knee_angle_left_total, knee_angle_right_total = 0, 0, 0, 0
        size = len(image_data_list)

        for shoulder_diff, hip_diff, knee_angle_left, knee_angle_right in image_data_list:
            shoulder_diff_total += shoulder_diff
            hip_diff_total += hip_diff
            knee_angle_left_total += knee_angle_left
            knee_angle_right_total += knee_angle_right

        if size:
            return (
                shoulder_diff_total / size,
                hip_diff_total / size,
                knee_angle_left_total / size,
                knee_angle_right_total / size
            )
        else:
            return None

    def angle_diff_to_muscle(self, shoulder_diff, hip_diff, knee_angle_left, knee_angle_right):
        """Map detected posture deviations to muscle weaknesses."""
        final_muscle_deficit = []
        final_joint_changes = []

        # Define angle changes and muscle groups
        angle_muscle_map = {
            "shoulder_imbalance": "latissimus dorsi, deltoids (anterior, lateral, posterior), teres minor, teres major, rhomboids, infraspinatus, subscapularis, upper trapezius, levator scapulae, supraspinatus, pectoralis major",
            "hip_imbalance": "gluteus maximus, gluteus medius, transversus abdominis, iliopsoas, hip flexors, adductors (adductor magnus, adductor longus), quadratus lumborum, rectus femoris, tensor fascia latae",
            "left_knee_valgus": "left gluteus medius, gluteus maximus, vastus medialis, hamstrings (biceps femoris), tibialis anterior, peroneus longus, gastrocnemius, soleus",
            "right_knee_valgus": "right gluteus medius, gluteus maximus, vastus medialis, hamstrings (biceps femoris), tibialis anterior, peroneus longus, gastrocnemius, soleus"
        }

        if shoulder_diff > 0.02:  
            final_muscle_deficit.append(angle_muscle_map["shoulder_imbalance"])
            final_joint_changes.append("Shoulder Asymmetry")

        if hip_diff > 0.02:  
            final_muscle_deficit.append(angle_muscle_map["hip_imbalance"])
            final_joint_changes.append("Hip Misalignment")

        if knee_angle_left < 165:  
            final_muscle_deficit.append(angle_muscle_map["left_knee_valgus"])
            final_joint_changes.append("Left Knee Valgus")

        if knee_angle_right < 165:  
            final_muscle_deficit.append(angle_muscle_map["right_knee_valgus"])
            final_joint_changes.append("Right Knee Valgus")

        return final_joint_changes, final_muscle_deficit



