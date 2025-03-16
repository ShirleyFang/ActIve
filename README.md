# ActIve
## Description
The **AI Physical Therapist** is an **on-device AI application** designed to help users improve their posture and prevent muscle imbalances through **real-time posture analysis** and **personalized exercise recommendations**. This application runs entirely on **Snapdragon X Elite Copilot+ PC**, ensuring **offline functionality and privacy protection**.

### Key Features
- **Posture Analysis (MediaPipe)**: Detects **postural imbalances** using real-time pose estimation from a webcam.
- **Wearable Data Integration**: Simulates real-world wearable device metrics (eventually we hope we can use real data as part of our application).
- **LLM-Powered Physical Therapy (Llama v3.1-8B-Chat)**: Generates **muscle imbalance priority ranking** and a **step-by-step corrective exercise plan**.
- **On-Device AI Processing**: Utilizes **ONNX Runtime & ollama** to run Llama models **fully offline**.
- **Privacy-Preserving & Edge AI**: No internet connection is required—**all user data remains local**.

### 🛠How It Works
**Step 1: Posture Detection** 📸
- The system captures **real-time images** of the user’s posture via a webcam and processes them using **MediaPipe Pose Estimation**.
- Identifies key postural issues such as **forward head posture, shoulder asymmetry, pelvic tilt, and knee misalignment**.
- Calculates **joint angles** to detect misalignment and muscle imbalances.
- Generates a **structured posture deviation report**.

**Step 2: Wearable Device Data Integrationy** 📊
- The system simulates **wearable device metrics** (or retrieves real-world sensor data) to **enhance posture analysis**.
- Factors considered: sleep duration, sitting duration, daily steps & activity levels, heart rate, etc.
- This data is **fed into a neural network (LifestyleNN)** that predicts **potential muscle weaknesses** based on lifestyle habits.

**Step 3: Muscle Weakness Priority Ranking (LLM)** 🏆  
- The **first LLM (Llama v3.1-8B-Chat)** takes the previous output and determines:  
- **Which muscle groups require the most attention** for rehabilitation.  
- **Priority ranking of muscles** that need targeted exercise for posture correction.
  
**Step 4: Personalized Exercise Plan Generation** 🏋️
- The **second LLM model** refines the **muscle group analysis** by integrating **user-specific data**:
- **User Profile** (age, gender, occupation, daily routine)
- **Training Preferences** (exercise frequency, available time, prior injuries)
- The model then generates a **step-by-step exercise plan**

## Dependencies:
- ONNXRuntime
- opencv-python         # OpenCV for computer vision tasks
- mediapipe             # MediaPipe for human pose estimation
- ollama                # Ollama CLI for AI model interaction


## How to Run:
1. Clone the repository:
    - git clone <repository-link>
2. Configure virual environment (Optional):
    - python -m venv venv
    - source venv/bin/activate
3. Install required dependencies:
    - pip3 install opencv-python mediapipe numpy
4. Run the program:
    - python3 XXXXX or python XXXXXX

### License:
    This project is for educational and research purposes only.
