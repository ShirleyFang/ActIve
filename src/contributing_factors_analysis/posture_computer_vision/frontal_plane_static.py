import cv2
import mediapipe as mp
import numpy as np
import os

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


def image_analysis(img_path):
    # Load image
    image_path = img_path  # Replace with your image path
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
        # 🟠 Hip Misalignment Detection
        hip_diff = abs(hip_left[1] - hip_right[1])
        # 🔴 Knee Valgus (Knock Knees) Detection
        knee_angle_left = calculate_angle(hip_left, knee_left, ankle_left)
        # 🔴 Knee Valgus (Knock Knees) Detection
        knee_angle_right = calculate_angle(hip_right, knee_right, ankle_right)

        return (shoulder_diff, hip_diff, knee_angle_left, knee_angle_right)
    else:
        print("No pose detected in the image.")
        return None


def image_process():
    image_path = "../data/img/user/"
    image_data_list = []

    for i in range(1):
        image_path += "image_" + str(i+1) + ".jpg"
        curr_result = image_analysis(image_path)
        if curr_result:
            image_data_list.append(curr_result)
        image_path = "../data/img/user/"

    return image_data_list


def average_image_data(image_data_list):
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


def angle_diff_to_muscle(shoulder_diff, hip_diff, knee_angle_left, knee_angle_right):
    final_muscle_deficit = []
    final_joint_changes = []

    # Define angle changes and muscle groups
    angle_muscle_map = {
        "shoulder_imbalance": "latissimus dorsi, deltoids (anterior, lateral, posterior), teres minor, teres major, rhomboids, infraspinatus, subscapularis, upper trapezius, levator scapulae, supraspinatus, pectoralis major",
        "hip_imbalance": "gluteus maximus, gluteus medius, transversus abdominis, iliopsoas, hip flexors, adductors (adductor magnus, adductor longus), quadratus lumborum, rectus femoris, tensor fascia latae",
        "left_knee_valgus": "left gluteus medius, gluteus maximus, vastus medialis, hamstrings (biceps femoris), tibialis anterior, peroneus longus, gastrocnemius, soleus",
        "right_knee_valgus": "right gluteus medius, gluteus maximus, vastus medialis, hamstrings (biceps femoris), tibialis anterior, peroneus longus, gastrocnemius, soleus"
    }

    if shoulder_diff > 0.02:  # Adjust threshold
        # print("⚠️ Warning: Shoulder Asymmetry Detected!")
        # print(angle_muscle_map["shoulder_imbalance"])
        final_muscle_deficit.append(angle_muscle_map["shoulder_imbalance"])
        final_joint_changes.append("Shoulder Asymmetry")

    if hip_diff > 0.02:  # Adjust threshold
        # print("⚠️ Warning: Hip Misalignment Detected!")
        # print(angle_muscle_map["hip_imbalance"])
        final_muscle_deficit.append(angle_muscle_map["hip_imbalance"])
        final_joint_changes.append("Hip Misalignment")

    if knee_angle_left < 165:  # Normal ~165-180°
        # print("⚠️ Warning: Left Knee Valgus (Inward Knees) Detected!")
        # print(angle_muscle_map["left_knee_valgus"])
        final_muscle_deficit.append(angle_muscle_map["left_knee_valgus"])
        final_joint_changes.append("Left Knee Valgus")

    if knee_angle_right < 165:  # Normal ~165-180°
        # print("⚠️ Warning: Right Knee Valgus (Inward Knees) Detected!")
        # print(angle_muscle_map["right_knee_valgus"])
        final_muscle_deficit.append(angle_muscle_map["right_knee_valgus"])
        final_joint_changes.append("Right Knee Valgus")

    # string_muscle_deficit = ("You are a physical therapist giving " +
    #                         "recommendations on a personal exercise " +
    #                         "program, please take into account that " +
    #                         "the user has the following muscle deficits: ")
    string_muscle_deficit = ", ".join(final_muscle_deficit)
    # string_joint_changes = ("You are a physical therapist giving " +
    #                         "recommendations on a personal exercise " +
    #                         "program, please take into account that " +
    #                         "the user has the following joint angle " +
    #                         "changes: ")
    string_joint_changes = ", ".join(final_joint_changes)

    print(string_joint_changes)
    print(string_muscle_deficit)


    # Show the image
    # cv2.imshow("Posture Analysis (Frontal Plane)", annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Draw landmarks and connections
    # annotated_image = image.copy()
    # mp_drawing.draw_landmarks(annotated_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
