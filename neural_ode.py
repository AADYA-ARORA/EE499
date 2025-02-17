import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchdiffeq import odeint
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Load dataset
data = pd.read_csv("battery_data.csv", delimiter=";")

# Select features and target
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

# Define the ODE Function
class BatteryODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, t, x):
        return self.fc2(self.relu(self.fc1(x)))

# Define the ODE Model
class ODEBatteryModel(nn.Module):
    def __init__(self, ode_func, input_dim):
        super().__init__()
        self.ode_func = ode_func
        self.linear = nn.Linear(input_dim, 1)  # Final layer to predict temperature
    
    def forward(self, x, t):
        out = odeint(self.ode_func, x, t)  # Solve the ODE
        return self.linear(out[-1])  # Take last time step output

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ode_func = BatteryODEFunc(input_dim=len(features)).to(device)
model = ODEBatteryModel(ode_func, input_dim=len(features)).to(device)

# Training Setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_ode_model(model, train_loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            t = torch.linspace(0, 1, steps=10).to(device)  # Time variable
            optimizer.zero_grad()
            y_pred = model(X_batch, t)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.5f}")

# Train Model
train_ode_model(model, train_loader)

# Evaluation function
def evaluate_ode_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            t = torch.linspace(0, 1, steps=10).to(device)
            y_pred = model(X_batch, t)
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(actuals)

# Evaluate Model
y_pred, y_actual = evaluate_ode_model(model, test_loader)

# Rescale predictions
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_actual_rescaled = scaler_y.inverse_transform(y_actual)

# Compute RMSE
rmse = np.sqrt(np.mean((y_pred_rescaled - y_actual_rescaled) ** 2))
print(f"Test RMSE: {rmse:.4f} °C")
