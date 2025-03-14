import cv2
import mediapipe as mp
import subprocess
from wearable_device.wearable_data import WearableSimulator # 引入可穿戴设备模拟数据

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 用户输入（提供默认值）
def get_user_input(prompt, default_value):
    user_input = input(f"{prompt} (useing default value using enter: {default_value}): ").strip()
    return user_input if user_input else default_value

user_gender = get_user_input("Please input gender: (Male/Female)", "Male")
user_age = get_user_input("Please input age: ", "30")
user_job = get_user_input("Please input your job: ", "Desk job")
user_habits = get_user_input("Is there any daily routine you wanna share? ", "Sits for 8 hours a day")

# 组合成 user_info
user_info = f"User is a {user_age}-year-old {user_gender}. They work as a {user_job}. Habits: {user_habits}."

# 读取本地图像
image_path = "../test.jpg"  # 确保 test.jpg 在项目根目录
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
prompt = f"User info: {user_info}\nPose analysis: {pose_description}\nWearable data: {wearable_description}\nProvide fitness and posture correction advice."


# 运行 Ollama 生成 AI 健身建议
response = subprocess.run(
    ["ollama", "run", "llama3", prompt], 
    capture_output=True, 
    text=True, 
    encoding="utf-8"
)

# 输出结果
print("\n========== AI Therapist Output ==========")
print("User Info:", user_info)
print("Pose Analysis:", pose_description)
print("AI Feedback:", response.stdout.strip())

# 输出错误信息（如果有）
if response.stderr:
    print("\nSTDERR:", response.stderr.strip())  
