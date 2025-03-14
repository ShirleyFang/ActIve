import cv2
import mediapipe as mp

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_pose(self, frame):
        """ 处理帧，检测人体姿势 """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results

    def draw_landmarks(self, frame, results):
        """ 在帧上绘制关键点 """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def extract_landmarks(self, results):
        """ 提取关键点数据 """
        if not results.pose_landmarks:
            return None
        landmarks = {idx: (lm.x, lm.y, lm.visibility) for idx, lm in enumerate(results.pose_landmarks.landmark)}
        return landmarks
