from user_info.user import User
from llm_model.ollama_model import OllamaModel

# Initialize user
user = User()
user_info = user.get_user_info()

pose_description = "Joint angles deviations based on posture analysis: anterior pelvic tilt and right shoulder elevation. Muscle imbalances to work on: rectus femoris, erector spinae, right lower trapezius, pectoralis minor."

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