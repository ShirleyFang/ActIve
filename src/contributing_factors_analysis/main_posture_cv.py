import os
import re
import socket

# Import the necessary modules (Assuming 'capture', 'fp', and 'sp' are classes you have)
# try:
#     socket.setdefaulttimeout(timeout=3)
#     socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
#     from image_capture import WebcamCapture  # Webcam image capture
#     from frontal_plane_static import FrontalPlanePostureAnalyzer  # Frontal posture analysis
#     from saggital_plane_static import SaggitalPlanePostureAnalyzer  # Saggital posture analysis
# except socket.error:
#     from contributing_factors_analysis.image_capture import WebcamCapture  # Webcam image capture
#     from contributing_factors_analysis.frontal_plane_static2 import FrontalPlanePostureAnalyzer  # Frontal posture analysis
#     from contributing_factors_analysis.saggital_plane_static2 import SaggitalPlanePostureAnalyzer  # Saggital posture analysis

# 
from contributing_factors_analysis.image_capture import WebcamCapture  # Webcam image capture
from contributing_factors_analysis.frontal_plane_static2 import FrontalPlanePostureAnalyzer  # Frontal posture analysis
from contributing_factors_analysis.saggital_plane_static2 import SaggitalPlanePostureAnalyzer  # Saggital posture analysis


class PostureAnalysis:
    CAMERA_MESSAGE = """
    📢 **Posture Analysis Notice**  

    You are about to take a **frontal standing picture** for analysis.  
    The system will automatically capture **10 images**, so please:  

    ✅ Maintain your posture **without moving** for accurate results.  
    ⚠️ **Any movement may affect the accuracy** of the analysis.  

    ⏳ **You have 20 seconds to prepare.**  
    Before the camera starts, a message will appear on the screen to notify you.  

    Thank you for your cooperation!  
    """

    def __init__(self, img_path="data/img/user/"):
        """Initialize posture analysis settings."""
        # load_dotenv()
        self.img_path = img_path
        self.capture = WebcamCapture()
        

    def start_frontal_posture_analysis(self):
        """Starts frontal posture analysis and returns detected issues."""
        print(self.CAMERA_MESSAGE)
        
        # Capture images using the external capture module
        self.capture.capture_images()

        # Create an instance of the FrontalPlanePostureAnalyzer class
        analyzer = FrontalPlanePostureAnalyzer()

        # Process images and analyze posture
        image_data_list = analyzer.image_process()

        average_result = analyzer.average_image_data(image_data_list)

        if average_result:
            (shoulder_diff, hip_diff, knee_angle_left, knee_angle_right) = average_result
            frontal_result = analyzer.angle_diff_to_muscle(shoulder_diff, hip_diff, knee_angle_left, knee_angle_right)
            print("Frontal posture analysis complete.")
        else:
            print("No result")
            frontal_result = None
        print(self.img_path)
        # Delete the image folder after analysis
        # self.capture.delete_folder()
        
        return frontal_result

    def start_saggital_posture_analysis(self):
        """Starts saggital posture analysis and returns detected issues."""
        print(self.CAMERA_MESSAGE)

        # Capture images using the external capture module
        # self.capture.capture_images()

        # Create an instance of the SaggitalPlanePostureAnalyzer class
        analyzer = SaggitalPlanePostureAnalyzer()

        # Process images and analyze posture
        image_data_list = analyzer.image_process()

        average_posture, hip_right, knee_right = analyzer.average_image_data(image_data_list)

        if average_posture:
            forward_head, rounded_shoulders, pelvic_tilt, knee_hyperextension = average_posture
            saggital_result = analyzer.postural_analysis(forward_head, rounded_shoulders, pelvic_tilt, knee_hyperextension, hip_right, knee_right)
            print("Saggital posture analysis complete.")
        else:
            print("No result")
            saggital_result = None

        # Delete the image folder after analysis
        # self.capture.delete_folder()

        return saggital_result


    def generate_posture_final_result(self):
        """Runs both frontal and saggital posture analyses and combines results."""
        final_deficit_muscles = set()
        final_joint_changes = set()
        
        frontal_joint_changes, frontal_muscle_deficit = self.start_frontal_posture_analysis()
        saggital_joint_changes, saggital_muscle_deficit = self.start_saggital_posture_analysis()
        final_deficit_muscles.update(self.remove_duplication(frontal_muscle_deficit))
        final_deficit_muscles.update(self.remove_duplication(saggital_muscle_deficit))

        final_joint_changes.update(self.remove_duplication(frontal_joint_changes))
        final_joint_changes.update(self.remove_duplication(saggital_joint_changes))

        string_joint_changes = ", ".join(final_joint_changes) if final_joint_changes else "No detected joint issues."
        string_muscle_deficit = ", ".join(final_deficit_muscles) if final_deficit_muscles else "No detected muscle weaknesses."
        print(string_muscle_deficit + " " + string_joint_changes)
        return string_muscle_deficit + " " + string_joint_changes

    @staticmethod
    def process_muscle_string(muscle_string, replacement_char="|"):
        """Process muscle string by preserving subcategories inside parentheses."""
        modified_string = re.sub(r"\(([^)]+)\)", lambda x: f"({x.group(1).replace(',', replacement_char)})", muscle_string)

        # Split by commas and store in a set
        muscle_list = {muscle.strip() for muscle in modified_string.split(",")}

        # Replace special character `|` back to commas inside `()`
        return {muscle.replace(replacement_char, ",") for muscle in muscle_list}

    def remove_duplication(self, original_list):
        """Remove duplicate muscle groups from the list."""
        result = set()
        for item in original_list:
            cleaned_items = self.process_muscle_string(item)
            result.update(cleaned_items)
        return result

# if __name__ == '__main__':
# analyzor = PostureAnalysis()
# analyzor.generate_posture_final_result()

