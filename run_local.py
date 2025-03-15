import cv2
import mediapipe as mp
from user_info.user import User
from models.ollama_model import OllamaModel

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize user
user = User()
user_info = user.get_user_info()

# 读取本地图像
image_path = "./test.jpg"  # 确保 test.jpg 在项目根目录
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 运行 MediaPipe 姿势检测
results = pose.process(image_rgb)

# 提取姿势关键点
if results.pose_landmarks:
    landmarks = {idx: (lm.x, lm.y, lm.visibility) for idx, lm in enumerate(results.pose_landmarks.landmark)}
    pose_description = "Joint angles deviations based on posture analysis: anterior pelvic tilt and right shoulder elevation. Muscle imbalances to work on: rectus femoris, erector spinae, right lower trapezius, pectoralis minor."
else:
    pose_description = "No pose detected."

# Initialize OllamaModel
ollama = OllamaModel()

# Output Results
print("\n========== AI Therapist Output ==========")
print("User Info:", user_info)
print("Pose Analysis:", pose_description)
print("\n🔥 **Step 1 - Muscle Distribution Analysis:**")

# 🔹 Step 1: AI analyzes muscle training distribution
muscle_distribution = ollama.analyze_muscle_distribution(pose_description)

print(muscle_distribution)  # Pretty print JSON
print("\n🏋️ **Step 2 - Final Training Plan:**")

# 🔹 Step 2: AI generates personalized training plan based on analysis
training_plan = ollama.generate_training_plan(user_info, muscle_distribution)

print(training_plan)