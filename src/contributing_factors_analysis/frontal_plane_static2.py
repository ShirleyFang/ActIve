import cv2
import numpy as np
import os
import onnxruntime as ort

# Define the absolute path to your ONNX model
MODEL_PATH = r"C:\Active_version2\ActIve\models\pose_landmarker_full.onnx"  # Update if needed

class FrontalPlanePostureAnalyzer:
    
    def __init__(self, model_path=MODEL_PATH):
        """Initialize ONNX Pose Estimation Model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])


        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])  # Change to GPU if needed
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_image(self, img_path):
        """Load and preprocess an image for ONNX inference."""
        if not os.path.exists(img_path):
            print(f"Error: Image not found at {img_path}")
            return None
        
        # Load image efficiently
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Unable to load image {img_path}")
            return None

        # Convert BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to match ONNX model input size (Update if needed)
        image_resized = cv2.resize(image_rgb, (128, 128), interpolation=cv2.INTER_AREA)

        # Normalize and reshape for ONNX
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_transposed = np.transpose(image_normalized, (2, 0, 1))  # Convert to (C, H, W)
        image_batch = np.expand_dims(image_transposed, axis=0)  # Add batch dimension

        return image_batch

    def process_image(self, img_path):
        """Run ONNX inference on an image."""
        input_data = self.preprocess_image(img_path)
        if input_data is None:
            return None

        # Run ONNX inference
        output = self.session.run([self.output_name], {self.input_name: input_data})
        
        # Extract pose keypoints from ONNX output
        keypoints = output[0][0]  # Modify based on actual ONNX model output format
        return keypoints

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a, b, c = np.array(a), np.array(b), np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        print(f"Angle: {angle}")
        return angle

    def image_analysis(self, img_path):
        """Analyze posture from an image using ONNX output."""
        keypoints = self.process_image(img_path)
        if keypoints is None:
            print("No pose detected.")
            return None

        # Extract key points based on model output
        shoulder_left = keypoints[11]
        shoulder_right = keypoints[12]
        hip_left = keypoints[23]
        hip_right = keypoints[24]
        knee_left = keypoints[25]
        knee_right = keypoints[26]
        ankle_left = keypoints[27]
        ankle_right = keypoints[28]

        # Compute posture deviations
        shoulder_diff = abs(shoulder_left[1] - shoulder_right[1])
        hip_diff = abs(hip_left[1] - hip_right[1])
        knee_angle_left = self.calculate_angle(hip_left, knee_left, ankle_left)
        knee_angle_right = self.calculate_angle(hip_right, knee_right, ankle_right)

        return (shoulder_diff, hip_diff, knee_angle_left, knee_angle_right)

    def image_process(self, image_folder="contributing_factors_analysis/user/"):
        """Process multiple images in a directory."""
        image_data_list = []
        image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]

        for img_file in image_files:
            full_image_path = os.path.join(image_folder, img_file)
            curr_result = self.image_analysis(full_image_path)
            if curr_result:
                image_data_list.append(curr_result)

        return image_data_list

    def average_image_data(self, image_data_list):
        """Compute the average values from multiple images."""
        if not image_data_list:
            return None

        avg_values = np.mean(image_data_list, axis=0)
        return tuple(avg_values)

    def angle_diff_to_muscle(self, shoulder_diff, hip_diff, knee_angle_left, knee_angle_right):
        """Map detected posture deviations to muscle weaknesses."""
        final_muscle_deficit = []
        final_joint_changes = []

        # Define muscle groups
        angle_muscle_map = {
            "shoulder_imbalance": "latissimus dorsi, deltoids, trapezius",
            "hip_imbalance": "gluteus maximus, iliopsoas, hip flexors",
            "left_knee_valgus": "left gluteus medius, hamstrings, tibialis anterior",
            "right_knee_valgus": "right gluteus medius, hamstrings, tibialis anterior"
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
        print(final_muscle_deficit)
        print(final_joint_changes)
        return final_joint_changes, final_muscle_deficit

# Run the class
# if __name__ == "__main__":
#     analyzer = FrontalPlanePostureAnalyzer()
#     image_data = analyzer.process_images()
    
#     if image_data:
#         avg_results = analyzer.compute_average_posture(image_data)
#         joint_changes, muscle_deficits = analyzer.angle_diff_to_muscle(*avg_results)

#         print("Posture Deviations:", joint_changes)
#         print("Associated Muscle Deficits:", muscle_deficits)
