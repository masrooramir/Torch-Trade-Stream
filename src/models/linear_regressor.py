# src/models/linear_regressor.py
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure the execution directory focuses correctly on the workspace root folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Select NVIDIA GPU - CUDA Acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearRegressionForecaster(nn.Module):
    """
    PyTorch Linear Regression Architecture mapping a 60-day sequence 
    input window down to 1 continuous target scalar prediction (Day 61).
    """
    def __init__(self, input_dim=60):
        super(LinearRegressionForecaster, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

def train_and_predict(ticker="AMZN"):
    print(f"\n🚀 Running Automated ML Engine for: {ticker} on device: {device}")
    
    # 1. Load Pre-computed Data Arrays from Disk
    try:
        X_train_raw = np.load(os.path.join(PROCESSED_DIR, f"{ticker}_X_train.npy"))
        y_train_raw = np.load(os.path.join(PROCESSED_DIR, f"{ticker}_y_train.npy"))
        scaler_params = np.load(os.path.join(PROCESSED_DIR, f"{ticker}_scaler_params.npy"))
    except FileNotFoundError as e:
        print(f"❌ Aborted: Pre-computed arrays missing for {ticker}. Ensure data pipeline ran first.")
        raise e

    # 2. Format into clean PyTorch Tensors
    X_train = torch.tensor(X_train_raw, dtype=torch.float32)
    y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(-1) # Shape: [Batch, 1]

    # Instantiate the DataLoader (Shuffle=True helps break temporal bias artifacts during gradient changes)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    # 3. Instantiate Network and Setup Optimization Layer
    model = LinearRegressionForecaster(input_dim=60).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Model Training Execution Loop
    model.train()
    for epoch in range(50):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Shift processing arrays directly onto NVIDIA CUDA cores
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward optimization tracking pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50] -> MSE Loss: {epoch_loss / len(train_loader):.6f}")

    # 5. Persist Trained Weights to Disk Storage
    weight_path = os.path.join(MODELS_DIR, f"{ticker}_linear_regressor.pth")
    torch.save(model.state_dict(), weight_path)
    print(f"✔ Trained model weights successfully persisted to: {weight_path}")

    # 6. Model Evaluation & Real Dollar Inverse-Scaling Transformation
    model.eval()
    with torch.no_grad():
        # Grab the absolute latest 60-day historical feature window to predict tomorrow
        latest_60_days = X_train[-1:].to(device) 
        predicted_61st_value = model(latest_60_days).item()
        print(f"🔮 Predicted scaled 61st value: {predicted_61st_value:.4f}")

        # Extract values using .item() to pull true numbers from multi-layered array vectors
        min_val = scaler_params[0].item()
        scale_val = scaler_params[1].item()

        # Apply the mathematical inverse transformation formula safely
        real_predicted_price = (predicted_61st_value - min_val) / scale_val
        print(f"💰 Decoded Predicted 61st Closing Price: ${real_predicted_price:.2f}")
        return real_predicted_price

if __name__ == "__main__":
    train_and_predict(ticker="AMZN")
