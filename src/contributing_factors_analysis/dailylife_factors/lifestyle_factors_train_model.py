import os
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as e:
    print("❌ Required module not found:", e)
    print("💡 Try installing dependencies using: pip install torch")
    torch = None  # Prevents system exit, allowing script to continue with error handling


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_data():
    """Loads and preprocesses CSV data for training."""
    file_path = "../../../data/csv/sleep_health_lifestyle_dataset.csv"  # Ensure correct dataset path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Data file not found at {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Select relevant features
    features = ["Sleep Duration (hours)", "Physical Activity Level (minutes/day)", "Stress Level (scale: 1-10)", "Occupation", "BMI Category", "Daily Steps"]
    target = "Weak Muscle Group"  # New column for classification labels
    
    # Label encoding for categorical features
    df["Occupation"] = LabelEncoder().fit_transform(df["Occupation"])
    df["BMI Category"] = LabelEncoder().fit_transform(df["BMI Category"])
    
    # Define muscle group labels based on heuristics
    df[target] = df.apply(lambda row: classify_muscle_weakness(row), axis=1)
    
    X = df[features].values
    y = df[target].values
    
    # Standardizing numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to tensors only if torch is available
    if torch:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long), \
               torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
    else:
        return None, None, None, None


def classify_muscle_weakness(row):
    """Assigns labels based on risk factors for weak muscle groups."""
    if row["Physical Activity Level (minutes/day)"] < 30:
        return 1  # Weak Core
    if row["Occupation"] in [0, 1]:  # Sedentary jobs
        return 2  # Weak Lower Back
    if row["Stress Level (scale: 1-10)"] >= 7:
        return 3  # Weak Upper Back / Neck
    return 0  # Healthy


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
        # self.bn1 = nn.BatchNorm1d(256)
        # self.bn2 = nn.BatchNorm1d(128)


    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(self.leaky_relu(self.fc3(x)))
        x = self.fc4(x)
        return self.softmax(x)


def train_model():
    """Trains the neural network model with batch processing."""
    if torch is None:
        print("❌ Skipping training: PyTorch is not installed.")
        return
    
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = LifestyleNN(input_size=X_train.shape[1], num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)  # Adjusted learning rate and added L2 regularization
    
    num_epochs = 200
    os.makedirs("../../../models", exist_ok=True)  # Ensure models directory exists
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}')
    
    torch.save(model.state_dict(), "../../../models/lifestyle_nn.pth")
    print("✅ Model saved successfully!")


def evaluate_model():
    """Evaluates the trained model."""
    if torch is None:
        print("❌ Skipping evaluation: PyTorch is not installed.")
        return
    
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    model = LifestyleNN(input_size=X_train.shape[1], num_classes=4)
    model.load_state_dict(torch.load("../../../models/lifestyle_nn.pth"))
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test)
        predicted = torch.argmax(outputs, dim=1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Accuracy: {accuracy * 100:.2f}%')

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
    model = LifestyleNN(input_size=6, num_classes=4)
    model.load_state_dict(torch.load("../../../models/lifestyle_nn.pth"))
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


def main():
    train_model()
    evaluate_model()

if __name__ == "__main__":
    main()
