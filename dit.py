import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Load dataset
data = pd.read_csv("battery_data.csv", delimiter=";")

# Select relevant features and target
features = ['v_raw_V', 'ocv_est_V', 'i_raw_A', 'soc_est', 'delta_q_Ah', 'EFC', 'cap_aged_est_Ah', 'R0_mOhm', 'R1_mOhm']
target = 't_cell_degC'

# Normalize data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
data[features] = scaler_x.fit_transform(data[features])
data[target] = scaler_y.fit_transform(data[[target]])

# Convert data to sequences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][features].values)
        y.append(data.iloc[i+seq_length][target])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(data, seq_length)
X_train, y_train = torch.tensor(X[:-100]), torch.tensor(y[:-100])
X_test, y_test = torch.tensor(X[-100:]), torch.tensor(y[-100:])

# Define PyTorch Dataset class
class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float().unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(BatteryDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(BatteryDataset(X_test, y_test), batch_size=32, shuffle=False)

# Define Diffusion Model Components
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * num_layers,
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x, noise_level):
        noise = torch.randn_like(x) * noise_level
        return self.model(x + noise)

# Initialize model
model = DiffusionModel(input_dim=len(features)).to("cuda" if torch.cuda.is_available() else "cpu")

# Training Setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_diffusion_model(model, train_loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            noise_level = torch.rand(1).to("cuda")
            optimizer.zero_grad()
            y_pred = model(X_batch, noise_level)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.5f}")

# Train the model
train_diffusion_model(model, train_loader)

# Testing function
def evaluate_diffusion_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            noise_level = torch.rand(1).to("cuda")
            y_pred = model(X_batch, noise_level)
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(actuals)

# Evaluate the model
y_pred, y_actual = evaluate_diffusion_model(model, test_loader)

# Rescale predictions
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_actual_rescaled = scaler_y.inverse_transform(y_actual)

# Compute RMSE
rmse = np.sqrt(np.mean((y_pred_rescaled - y_actual_rescaled) ** 2))
print(f"Test RMSE: {rmse:.4f} °C")
