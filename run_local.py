import cv2
import mediapipe as mp
from user_info.user import User
from wearable_device.wearable_data import WearableSimulator
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
    pose_description = f"User has a forward-leaning posture with hip at {landmarks.get(23, 'N/A')} and knee at {landmarks.get(25, 'N/A')}"
else:
    pose_description = "No pose detected."

# 获取模拟的可穿戴设备数据
wearable = WearableSimulator()
wearable_data = wearable.get_data()
wearable_description = f"Sedentary for {wearable_data['sedentary_hours']} hours per day. Avg HR: {wearable_data['avg_heart_rate']} bpm. Peak HR: {wearable_data['peak_heart_rate']} bpm."

# 生成 Prompt
# base info + bethshirley analysize + basic prompt
prompt = f"User info: {user_info}\nPose analysis: {pose_description}\nWearable data: {wearable_description}\nProvide fitness and posture correction advice."


# Initialize OllamaModel
ollama = OllamaModel()

# Generate AI feedback using the refactored class
ai_feedback = ollama.generate_feedback(user_info, pose_description)

# Output Results
print("\n========== AI Therapist Output ==========")
print("User Info:", user_info)
print("Pose Analysis:", pose_description)
print("AI Feedback:", ai_feedback)