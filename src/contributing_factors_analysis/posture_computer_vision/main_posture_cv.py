import image_capture as capture
import os
from dotenv import load_dotenv
import frontal_plane_static as fp


def start_posture_analysis():
    load_dotenv()
    # img_path = os.getenv("IMG_PATH")
    img_path = "../data/img/user/"
    capture.capture_images()
    # fp.image_process()

    # image_data_list = fp.image_process()
    # print(f"image data: {image_data_list}")
    # average_result = fp.average_image_data(image_data_list)
    # print(f"avg result: {average_result}")

    # if average_result:
    #     (shoulder_diff, hip_diff, knee_angle_left, knee_angle_right) = average_result
    #     fp.angle_diff_to_muscle(shoulder_diff, hip_diff, knee_angle_left, knee_angle_right)
    #     print("analyzed result")
    # else:
    #     print("No result")

    # capture.delete_folder(img_path)


start_posture_analysis()
