class User:
    def __init__(self):
        """Initialize user with default values and collect input."""
        print("\n Press feel free to skip any of the questions if you don't wanna answer. \n")
        self.user_gender = self.get_user_input("Please input gender: ", "")
        self.user_age = self.get_user_input("Please input age: ", "")
        self.user_job = self.get_user_input("Please input your job: ", "")
        self.user_habits = self.get_user_input("Is there any daily routine you wanna share? ", "")
        self.exercise_days = self.get_user_input("How many days you wanna exerceis in a week? ", "")
        self.exercise_time = self.get_user_input("How much time you can exerceis a day? ", "")
        self.extra_info = self.get_user_input("Is there any extra information you wanna share? ", "")

    @staticmethod
    def get_user_input(prompt, default_value):
        """Helper function to get user input with default values."""
        user_input = input(f"{prompt} (Press enter to skip {default_value})").strip()
        return user_input if user_input else default_value

    def get_user_info(self):
        """Return formatted user information."""
        return f"""
        - User is a {self.user_age}-year-old {self.user_gender}. 
        - They work as a {self.user_job}. 
        - Habits: {self.user_habits}.
        - The user can exercise {self.exercise_days} days, and {self.exercise_time} a time.
        - Extra information: {self.extra_info}
        """

# Usage Example:
# user = User()
# print(user.get_user_info())
