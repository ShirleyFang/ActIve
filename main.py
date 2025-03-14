from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from pose_detection.pose_estimator import PoseEstimator
from models.ollama_model import OllamaModel

# 初始化 Flask 应用
app = Flask(__name__)

# 初始化 MediaPipe 姿势检测 & Ollama LLM
pose_estimator = PoseEstimator()
ollama_model = OllamaModel()

@app.route("/")
def index():
    """ 加载前端页面 """
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_pose():
    """ 处理前端上传的帧图像，分析用户姿势并生成 AI 反馈 """
    if "video_frame" not in request.files:
        return jsonify({"error": "No video frame uploaded"}), 400

    file = request.files["video_frame"]
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # 使用 MediaPipe 进行姿势检测
    results = pose_estimator.detect_pose(frame)
    landmarks = pose_estimator.extract_landmarks(results)
    
    if landmarks:
        pose_description = f"User has a forward-leaning posture with hip at {landmarks.get(23, 'N/A')} and knee at {landmarks.get(25, 'N/A')}"
    else:
        pose_description = "No pose detected."

    # 获取用户输入的信息（例如年龄、职业等）
    user_info = request.form.get("user_info", "User information not provided.")

    # 生成 AI 反馈（调用 Ollama LLM）
    feedback = ollama_model.generate_feedback(user_info, pose_description)

    return jsonify({
        "pose": pose_description,
        "feedback": feedback
    })

if __name__ == "__main__":
    app.run(debug=True)
