import os
import torch
import torch.nn as nn
# Get the absolute path of the contributing_factors_analysis directory

# Now import the module
# import posture_computer_vision.image_capture as capture
import numpy as np
from sklearn.preprocessing import StandardScaler
from contributing_factors_analysis.main_posture_cv import PostureAnalysis
from contributing_factors_analysis.dailylife_data_simulator import WearableSimulator


# Import the necessary modules (Assuming 'capture', 'fp', and 'sp' are classes you have)

class LifestyleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LifestyleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.4)  # Increased dropout for better generalization
        self.fc4 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(self.leaky_relu(self.fc3(x)))
        x = self.fc4(x)
        return self.softmax(x)

def analyze_lifestyle(user_data):
    """Analyzes user lifestyle data and predicts muscle weakness based on trained model."""

    # ✅ Define scaler with the same parameters used during training
    scaler = StandardScaler()
    scaler.mean_ = np.array([7, 45, 5, 1, 1, 5000])  # Example mean values
    scaler.scale_ = np.array([1.5, 30, 2, 1, 1, 3000])  # Example std dev values

    # ✅ Convert user input to a NumPy array and apply the trained scaler
    user_data_np = np.array(user_data).reshape(1, -1)
    user_data_scaled = scaler.transform(user_data_np)

    # ✅ Load the trained model

    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    github_final_dir = os.path.dirname(src_dir)
    model_path = os.path.join(github_final_dir, "models", "lifestyle_nn.pth")
    model_path = os.path.normpath(model_path)

    model = LifestyleNN(input_size=6, num_classes=4)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # ✅ Convert to tensor and make predictions
    user_tensor = torch.tensor(user_data_scaled, dtype=torch.float32)
    with torch.no_grad():
        output = model(user_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # ✅ Muscle weakness labels
    weakness_labels = ["Healthy", "Weak Core", "Weak Lower Back", "Weak Upper Back / Neck"]
    print(f"\n🔍 Analysis Result: {weakness_labels[predicted_class]} Muscle Weakness Detected!\n")

    return weakness_labels[predicted_class]


def retrieve_final_result():
    # wearable = WearableSimulator()

    # Get and print wearable device data
    # data = wearable.get_data()
    postureAnalysis = PostureAnalysis()
    result1 = ""
    # result1 = analyze_lifestyle(data)
    result2 = postureAnalysis.generate_posture_final_result()
    return result1 + " " + result2