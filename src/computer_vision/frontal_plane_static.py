import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Middle joint
    c = np.array(c)  # Last joint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    print(f"Angle: {angle}")
    return angle


final_muscle_deficit = []
final_joint_changes = []

# Define angle changes and muscle groups
angle_muscle_map = {
    "shoulder_imbalance": "latissimus dorsi",
    "hip_imbalance": "gluteus maximus, transversus abdominis",
    "left_knee_valgus": "left gluteus medius, gluteus maximus",
    "right_knee_valgus": "right gluteus medius, gluteus maximus"
}

# Load image
image_path = "knee_valgus.png"  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image with MediaPipe
result = pose.process(image_rgb)

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
    if shoulder_diff > 0.02:  # Adjust threshold
        # print("⚠️ Warning: Shoulder Asymmetry Detected!")
        # print(angle_muscle_map["shoulder_imbalance"])
        final_muscle_deficit.append(angle_muscle_map["shoulder_imbalance"])
        final_joint_changes.append("Shoulder Asymmetry")

    # 🟠 Hip Misalignment Detection
    hip_diff = abs(hip_left[1] - hip_right[1])
    if hip_diff > 0.02:  # Adjust threshold
        # print("⚠️ Warning: Hip Misalignment Detected!")
        # print(angle_muscle_map["hip_imbalance"])
        final_muscle_deficit.append(angle_muscle_map["hip_imbalance"])
        final_joint_changes.append("Hip Misalignment")

    # 🔴 Knee Valgus (Knock Knees) Detection
    knee_angle_left = calculate_angle(hip_left, knee_left, ankle_left)

    if knee_angle_left < 165:  # Normal ~165-180°
        # print("⚠️ Warning: Left Knee Valgus (Inward Knees) Detected!")
        # print(angle_muscle_map["left_knee_valgus"])
        final_muscle_deficit.append(angle_muscle_map["left_knee_valgus"])
        final_joint_changes.append("Left Knee Valgus")

    # 🔴 Knee Valgus (Knock Knees) Detection
    knee_angle_right = calculate_angle(hip_right, knee_right, ankle_right)
    if knee_angle_right < 165:  # Normal ~165-180°
        # print("⚠️ Warning: Right Knee Valgus (Inward Knees) Detected!")
        # print(angle_muscle_map["right_knee_valgus"])
        final_muscle_deficit.append(angle_muscle_map["right_knee_valgus"])
        final_joint_changes.append("Right Knee Valgus")

    string_muscle_deficit = ("You are a physical therapist giving " +
                             "recommendations on a personal exercise " +
                             "program, please take into account that " +
                             "the user has the following muscle deficits: ")
    string_muscle_deficit += ", ".join(final_muscle_deficit)
    string_joint_changes = ("You are a physical therapist giving " +
                            "recommendations on a personal exercise " +
                            "program, please take into account that " +
                            "the user has the following joint angle " +
                            "changes: ")
    string_joint_changes += ", ".join(final_joint_changes)

    print(string_joint_changes)
    print(string_muscle_deficit)

    # Draw landmarks and connections
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the image
    cv2.imshow("Posture Analysis (Frontal Plane)", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No pose detected in the image.")
