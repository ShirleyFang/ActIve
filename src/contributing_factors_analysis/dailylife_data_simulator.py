import random

class WearableSimulator:

    def __init__(self):
        """Initialize a simulator to generate realistic user lifestyle data."""
        self.sleep_duration = random.uniform(4.5, 9)  # Sleep (4.5 to 9 hours)
        self.physical_activity = random.randint(0, 120)  # Physical activity (0 to 120 mins)
        self.stress_level = random.randint(1, 10)  # Stress level (1 to 10)
        self.occupation = random.choice([0, 1, 2])  # 0: Sedentary, 1: Active, 2: Highly Active
        self.bmi_category = random.choice([0, 1, 2, 3])  # 0: Underweight, 1: Normal, 2: Overweight, 3: Obese
        self.daily_steps = random.randint(1000, 15000)  # Steps per day (1,000 to 15,000)

    def get_data(self):
        """Returns a simulated user dataset formatted for analysis."""
        return [
            round(self.sleep_duration, 1),
            self.physical_activity,
            self.stress_level,
            self.occupation,
            self.bmi_category,
            self.daily_steps
        ]


# # 测试代码
# if __name__ == "__main__":
#     wearable = WearableSimulator()
    