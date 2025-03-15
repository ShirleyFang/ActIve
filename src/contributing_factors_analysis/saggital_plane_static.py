import cv2
import mediapipe as mp
import numpy as np
import os
# import image_capture as capture  # Webcam image capture


class SaggitalPlanePostureAnalyzer:
    def __init__(self, num_photos=1, image_path="user/"):
        """Initialize MediaPipe Pose Estimation and Image Processing Configs."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.num_photos = num_photos
        self.image_path = image_path

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
        """Analyze the image and extract posture key points."""
        if not os.path.exists(img_path):
            print(f"Error: Image path {img_path} does not exist.")
            return None

        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Unable to load image at {img_path}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.pose.process(image_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Extract key points
            head = [landmarks[0].x, landmarks[0].y]
            shoulder_right = [landmarks[12].x, landmarks[12].y]
            hip_right = [landmarks[24].x, landmarks[24].y]
            knee_right = [landmarks[26].x, landmarks[26].y]
            ankle_right = [landmarks[28].x, landmarks[28].y]

            # Postural Deviations
            forward_head = abs(head[0] - shoulder_right[0]) > 0.02
            rounded_shoulders = abs(shoulder_right[0] - head[0]) > 0.02
            pelvic_tilt = abs(hip_right[1] - knee_right[1]) > 0.02
            knee_angle = self.calculate_angle(hip_right, knee_right, ankle_right)
            knee_hyperextension = knee_angle < 170

            return forward_head, rounded_shoulders, pelvic_tilt, knee_hyperextension, hip_right, knee_right
        else:
            print("No pose detected in the image.")
            return None

    def image_process(self):
        """Process all images and collect posture data."""
        image_data_list = []

        for i in range(self.num_photos):
            full_path = os.path.join(self.image_path, f"image_{i+1}.jpg")
            curr_result = self.image_analysis(full_path)
            if curr_result:
                image_data_list.append(curr_result)
            else:
                print(f"Skipping image {full_path} due to errors.")

        return image_data_list

    def average_image_data(self, image_data_list):
        """Compute average postural data from multiple images."""
        if not image_data_list:
            return None

        forward_head_total, rounded_shoulders_total, pelvic_tilt_total, knee_hyperextension_total = 0, 0, 0, 0
        size = len(image_data_list)

        for data in image_data_list:
            forward_head, rounded_shoulders, pelvic_tilt, knee_hyperextension, _, _ = data
            forward_head_total += forward_head
            rounded_shoulders_total += rounded_shoulders
            pelvic_tilt_total += pelvic_tilt
            knee_hyperextension_total += knee_hyperextension

        return (
            forward_head_total / size,
            rounded_shoulders_total / size,
            pelvic_tilt_total / size,
            knee_hyperextension_total / size
        ), image_data_list[0][4], image_data_list[0][5]  # Return hip_right and knee_right too

    def recommend_muscles(self, forward_head, rounded_shoulders, pelvic_tilt, knee_hyperextension, hip_right, knee_right):
        """Recommend muscles to strengthen based on detected postural issues."""
        muscles_to_strengthen = []

        angle_muscle_map = {
            "forward_head_muscles": "Deep Neck Flexors (longus colli, longus capitis), Upper Back (rhomboids, middle trapezius)",
            "rounded_shoulders": "Lower Trapezius, Rhomboids, Rotator Cuff Muscles",
            "anterior_pelvic_tilt": "Glutes and Hamstrings, Abdominals",
            "posterior_pelvic_tilt": "Hip flexors, Lower Back Muscles (erector spinae)",
            "knee_hyperextension": "Hamstrings, Quadriceps, Hip Flexors, Gluteus Maximus",
        }

        if forward_head:
            muscles_to_strengthen.append(angle_muscle_map["forward_head_muscles"])

        if rounded_shoulders:
            muscles_to_strengthen.append(angle_muscle_map["rounded_shoulders"])

        if pelvic_tilt:
            if hip_right[1] < knee_right[1]:  # Anterior Pelvic Tilt
                muscles_to_strengthen.append(angle_muscle_map["anterior_pelvic_tilt"])
            else:  # Posterior Pelvic Tilt
                muscles_to_strengthen.append(angle_muscle_map["posterior_pelvic_tilt"])

        if knee_hyperextension:
            muscles_to_strengthen.append(angle_muscle_map["knee_hyperextension"])

        return muscles_to_strengthen

    def postural_analysis(self, forward_head, rounded_shoulders, pelvic_tilt, knee_hyperextension, hip_right, knee_right):
        """Analyze postural issues and provide recommendations."""
        final_issues = []

        if forward_head:
            final_issues.append("Forward Head Posture Detected")
        if rounded_shoulders:
            final_issues.append("Rounded Shoulders Detected")
        if pelvic_tilt:
            final_issues.append("Pelvic Tilt (Anterior or Posterior) Detected")
        if knee_hyperextension:
            final_issues.append("Knee Hyperextension Detected")

        # Pass hip_right and knee_right to the function
        muscles_to_strengthen = self.recommend_muscles(forward_head, rounded_shoulders, pelvic_tilt, knee_hyperextension, hip_right, knee_right)

        if final_issues:
            string_joint_changes = ", ".join(final_issues)
            print(string_joint_changes)
        else:
            string_joint_changes = "No significant postural deviations detected."
            print(string_joint_changes)

        if muscles_to_strengthen:
            string_muscle_deficit = ", ".join(muscles_to_strengthen)
            print(string_muscle_deficit)
        else:
            string_muscle_deficit = "No significant muscle imbalances detected."

        return final_issues, muscles_to_strengthen

