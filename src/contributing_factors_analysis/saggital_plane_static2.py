import cv2
import numpy as np
import os
import onnxruntime as ort

# Define the absolute path to your ONNX model
MODEL_PATH = r"C:\Active_version2\ActIve\models\pose_landmarker_full.onnx"  # Update this path if needed

class SaggitalPlanePostureAnalyzer:
    
    def __init__(self, num_photos=1, image_path="contributing_factors_analysis/user/"):
        """Initialize ONNX Pose Estimation Model and Image Processing Configs."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        # self.session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider'])


        self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])  # Change to GPU if needed
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.num_photos = num_photos
        self.image_path = os.path.abspath(image_path)  # Ensure correct path format

    def preprocess_image(self, img_path):
        """Load and preprocess an image for ONNX inference."""
        if not os.path.exists(img_path):
            print(f"⚠️ Error: Image not found at {img_path}")
            return None
        
        # Load image efficiently
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"⚠️ Error: Unable to load image {img_path}")
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
        return angle

    def image_analysis(self, img_path):
        """Analyze the image and extract posture key points."""
        keypoints = self.process_image(img_path)
        if keypoints is None:
            print("⚠️ No pose detected.")
            return None

        # Extract key points based on ONNX model output format
        head = keypoints[0]
        shoulder_right = keypoints[12]
        hip_right = keypoints[24]
        knee_right = keypoints[26]
        ankle_right = keypoints[28]

        # Postural Deviations
        forward_head = abs(head[0] - shoulder_right[0]) > 0.02
        rounded_shoulders = abs(shoulder_right[0] - head[0]) > 0.02
        pelvic_tilt = abs(hip_right[1] - knee_right[1]) > 0.02
        knee_angle = self.calculate_angle(hip_right, knee_right, ankle_right)
        knee_hyperextension = knee_angle < 170

        return forward_head, rounded_shoulders, pelvic_tilt, knee_hyperextension, hip_right, knee_right

    def image_process(self):
        """Process all images and collect posture data."""
        image_data_list = []

        # ✅ Ensure the path is correctly built
        full_path = os.path.abspath(os.path.join(self.image_path, "image_2.jpg"))

        # ✅ Print the corrected path
        print(f"📸 Processing Image: {full_path}")

        if not os.path.exists(full_path):
            print(f"⚠️ Error: Image not found at {full_path}")
            return None

        curr_result = self.image_analysis(full_path)

        if curr_result:
            image_data_list.append(curr_result)
        else:
            print(f"⚠️ Skipping Image: {full_path}")

        return image_data_list if image_data_list else None

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
        ), image_data_list[0][4], image_data_list[0][5]

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
            final_issues.append("Pelvic Tilt Detected")
        if knee_hyperextension:
            final_issues.append("Knee Hyperextension Detected")

        muscles_to_strengthen = self.recommend_muscles(forward_head, rounded_shoulders, pelvic_tilt, knee_hyperextension, hip_right, knee_right)
        return final_issues, muscles_to_strengthen
