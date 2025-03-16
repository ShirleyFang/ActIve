import subprocess

class OllamaModel:
    """Handles AI-generated feedback using Ollama in two-step processing."""

    def __init__(self, model_name="llama3.2:1b"):
        """
        Initialize the model.
        - Ensures the model is preloaded before any requests.
        """
        self.model_name = model_name
        print(f"✅ Model {self.model_name} loaded successfully!")
        # self._ensure_model_loaded()

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

    def analyze_muscle_distribution(self, pose_description):
        """
        Step 1: Analyze posture deviations and determine the priority of muscle groups.
        
        Returns:
            string containing muscle training priority analysis.
        """
        prompt = f"""
        You are an expert physical therapist. Given{pose_description} Analyze the user's posture deviations and prioritize the Top3 muscle groups that need training.
        Return ONLY the following format (no extra text):

        🔹 **Muscle Priority Analysis:**
            - Prioritize top3 muscle group based on its importance in posture correction.
            - Example Output Format(the order shows priority):
                1. gluteus_maximus
                2. transversus_abdominis
                3. left_gluteus_medius
        """

        response = self._generate_response(prompt)
        
        return response

    def generate_training_plan(self, user_info, muscle_distribution):
        """
        Step 2: Generate a personalized training plan based on muscle group analysis.
        """
        prompt = f"""
        You are a professional physical therapist. Based on the user's posture analysis and optimal muscle training ratio, create a **detailed** personalized exercise plan.
        Here's the information you need, user_info: {user_info} and muscle_analysis {muscle_distribution}.

        🔹 **Exercise Plan Requirements:**
            - Assign **specific exercises** for each muscle group.
            - Define **sets, reps, and rest intervals**.
            - Suggest **corrective posture tips** to integrate into daily life.
            - Provide a **step-by-step structured response**.

        🔹 **Exercise Plan** 
        ```
            Day 1:
        1️⃣ **Gluteus Maximus**
           - **Exercise:** Hip Thrusts
           - **Sets & Reps:** 4 sets × 12 reps
           - **Rest Time:** 60 seconds
           - **Correction Tip:** Engage core to stabilize pelvis.

        2️⃣ **Transversus Abdominis**
           - **Exercise:** Dead Bugs
           - **Sets & Reps:** 3 sets × 15 reps
           - **Rest Time:** 45 seconds
           - **Correction Tip:** Keep lower back neutral.
        
        ```
        """
        return self._generate_response(prompt)

    def _generate_response(self, prompt):
        """Handles interaction with Ollama."""
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







