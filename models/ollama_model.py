import subprocess

class OllamaModel:
    """Handles AI-generated feedback using Ollama."""

    def __init__(self, model_name="llama3.1:8b"):
        """
        Initialize the model.
        - Ensures the model is preloaded before any requests.
        """
        self.model_name = model_name
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """
        Ensure that the model is available in Ollama before generating responses.
        """
        try:
            print(f"🔄 Loading model: {self.model_name}...")
            subprocess.run(["ollama", "pull", self.model_name], check=True, capture_output=True)
            print(f"✅ Model {self.model_name} loaded successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to load model {self.model_name}: {e}")

    def generate_feedback(self, user_info, pose_description):
        """
        Generates AI feedback based on user information and pose analysis.
        - Trims overly long prompts to optimize response time.
        - Limits response tokens to speed up inference.

        Args:
            user_info (str): Formatted user description.
            pose_description (str): Analysis from MediaPipe.
            max_tokens (int): Limits token output to speed up response time.

        Returns:
            str: AI-generated feedback.
        """
        # Reduce prompt length for faster execution

        prompt = (
            f"You are an expert physical therapist. A user has shared their posture and fitness details with you.\n"
            f"Your response should be structured as follows:\n\n"
            f"1️⃣ **Posture Analysis:**\n"
            f"   - Identify potential muscle weaknesses or imbalances.\n"
            f"   - Explain why poor posture occurs in this case.\n\n"
            f"2️⃣ **Muscle & Joint Impact:**\n"
            f"   - Discuss which muscle groups are affected (e.g., weak core, tight hamstrings).\n"
            f"   - Provide scientific reasoning for why these issues arise.\n\n"
            f"3️⃣ **Personalized Corrective Plan:**\n"
            f"   - Suggest a **detailed** corrective exercise plan.\n"
            f"   - Include sets, reps, and frequency.\n\n"
            f"4️⃣ **Posture & Habit Correction:**\n"
            f"   - Provide actionable tips on how the user can improve their daily routine.\n"
            f"   - Suggest ergonomic improvements for their workspace.\n\n"
            f"5️⃣ **Expected Results & Timeline:**\n"
            f"   - Explain how long it may take to see improvements.\n"
            f"   - Set realistic expectations.\n\n"
            f"User Info:\n{user_info}\n\n"
            f"Pose Analysis:\n{pose_description}\n\n"
        )
        try:
            response = subprocess.run(
                ["ollama", "run", self.model_name, prompt], 
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            return response.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"❌ AI Generation Error: {e}"


