import subprocess
import requests

class OllamaModel:
    """Handles AI-generated feedback using Ollama in two-step processing."""

    def __init__(self, model_name="llama3.2:1b"):
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

    def analyze_muscle_distribution(self, pose_description):
        """
        Step 1: Analyze posture deviations and determine the priority of muscle groups.
        
        Returns:
            string containing muscle training priority analysis.
        """
        prompt = f"""
        You are an expert physical therapist. Analyze the user's posture deviations and prioritize the muscle groups that need training.
        The outout should ONLY include the following parts:

        🔹 **Posture Analysis Report:**
            - {pose_description}

        🔹 **Muscle Priority Analysis:**
            - Prioritize each muscle group based on its importance in posture correction with explanizations.
            - Example Output Format(the order shows priority):
                1. gluteus_maximus
                2. transversus_abdominis
                3. left_gluteus_medius
                4. hip_flexors
        """

        response = self._generate_response(prompt)
        
        return response

    def generate_training_plan(self, user_info, muscle_distribution):
        """
        Step 2: Generate a personalized training plan based on muscle group analysis.
        """
        prompt = f"""
        You are a professional physical therapist designing a structured, **time-efficient** training plan. Your goal is to create a well-balanced routine **without splitting by days (NO Day 1, 2, 3 structure)**.

        **User Profile:** {user_info}
        **Muscle Weakness Analysis:** {muscle_distribution}

        🔹 **Exercise Plan Structure:**
        - **Each muscle group** must have at least one assigned exercise.
        - **Clearly specify** the **exercise name**, **sets, reps, rest time**, and **estimated duration** per exercise.
        - **Total workout time should be under 60 minutes.**
        - **NO "Day 1, Day 2, etc."** – just a complete list of exercises with estimated time.

        🔹 **Example Output Format:**
        ```
        1️⃣ **Gluteus Maximus**
        - **Exercise:** Hip Thrusts
        - **Sets & Reps:** 4 sets × 12 reps
        - **Rest Time:** 60 seconds
        - **Estimated Time:** 8 minutes
        - **Correction Tip:** Engage core to stabilize pelvis.

        2️⃣ **Transversus Abdominis**
        - **Exercise:** Dead Bugs
        - **Sets & Reps:** 3 sets × 15 reps
        - **Rest Time:** 45 seconds
        - **Estimated Time:** 5 minutes
        - **Correction Tip:** Keep lower back neutral.
        ```

        🔹 **Important Instructions for AI:**
        - **DO NOT group by "Day 1, Day 2"** – list all exercises together.
        - **Include estimated time per exercise**.
        - **Ensure total workout time does not exceed 60 minutes**.
        - **Provide correction tips for each exercise**.

        Now, based on the given user profile and muscle weakness, generate a structured workout plan following these rules.
        """
        return self._generate_response(prompt)

    #     """
    #     Step 2: Generate a personalized training plan based on muscle group analysis.
    #     """
    #     prompt = f"""
    #     You are a professional physical therapist. Based on the user's posture analysis and optimal muscle training ratio, create a **detailed** personalized exercise plan.
    #     Here's the information you need, user_info: {user_info} and muscle_analysis {muscle_distribution}.

    #     🔹 **Exercise Plan Requirements:**
    #         - Assign **specific exercises** for each muscle group.
    #         - Define **sets, reps, and rest intervals**.
    #         - Suggest **corrective posture tips** to integrate into daily life.
    #         - Provide a **step-by-step structured response**.

    #     🔹 **Exercise Plan** 
    #     ```
    #     1️⃣ **Gluteus Maximus**
    #        - **Exercise:** Hip Thrusts
    #        - **Sets & Reps:** 4 sets × 12 reps
    #        - **Rest Time:** 60 seconds
    #        - **Correction Tip:** Engage core to stabilize pelvis.

    #     2️⃣ **Transversus Abdominis**
    #        - **Exercise:** Dead Bugs
    #        - **Sets & Reps:** 3 sets × 15 reps
    #        - **Rest Time:** 45 seconds
    #        - **Correction Tip:** Keep lower back neutral.
    #     ```
    #     """
    #     return self._generate_response(prompt)



    def _generate_response(self, prompt):
        """Handles interaction with Ollama."""
        try:
            # response = subprocess.run(
            #     ["ollama", "run", self.model_name, prompt],
            #     capture_output=True,
            #     text=True,
            #     encoding="utf-8"
            # )
            
            # return response.stdout.strip()
            response = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            
            return response.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"❌ AI Generation Error: {e}"
        






