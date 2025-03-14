import subprocess

class OllamaModel:
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name

    def generate_feedback(self, user_input, pose_description):
        """ 调用 Ollama 生成 AI 健身建议 """
        prompt = f"""User info: {user_input}\nPose analysis: {pose_description}\n
        Provide detailed exercise plan. 
        For example, there would be 20min for muscle 1, and there would be 3 movements in this 20min, 
        exaplain which movements works on which muscle, and give instructions on this movement.
        Organize it in a nice formate."""
        response = subprocess.run(["ollama", "run", self.model_name, prompt], capture_output=True, text=True)
        return response.stdout.strip()
