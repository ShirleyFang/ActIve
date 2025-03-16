from user_info.user import User
from llm_model.ollama_model import OllamaModel
from contributing_factors_analysis.combine_daliylife_and_cv import retrieve_final_result

# Initialize user
user = User()
user_info = user.get_user_info()

# Initialize and get data from CV model
pose_description = retrieve_final_result()

# Initialize llama Model
ollama = OllamaModel()

# Output Results
print("\n========== AI Therapist Output ==========")
print("User Info:", user_info)
# print("Pose Analysis:", pose_description)
print("\n🔥 **Step 1 - Muscle Priority Analysis:**")

# 🔹 Step 1: AI analyzes muscle training distribution
muscle_distribution = ollama.analyze_muscle_distribution(pose_description)

print(muscle_distribution)  # Pretty print JSON
print("\n🏋️ **Step 2 - Final Training Plan:**")

# 🔹 Step 2: AI generates personalized training plan based on analysis
training_plan = ollama.generate_training_plan(user_info, muscle_distribution)

print(training_plan)

