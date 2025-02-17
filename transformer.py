import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# Load dataset

data = pd.read_csv("/Users/aadya/Downloads/EE499/battery_dataset1.csv", delimiter=";")  # Specify semicolon as the separator
  # Ensure your dataset is named correctly
print(data.columns)  # Print the column names to check if they match
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
print(data.head())  # Check if data is loaded correctly
features = list(data.columns)  # Use actual column names from CSV

# Select relevant features and target
features = ["timestamp_s","v_raw_V","ocv_est_V","i_raw_A","soc_est","delta_q_Ah","EFC","cap_aged_est_Ah","R0_mOhm","R1_mOhm"]
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

seq_length = 10  # Define sequence length
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

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])  # Use last time step for prediction
        return x

# Initialize model
model = TransformerModel(input_dim=len(features)).to("cuda" if torch.cuda.is_available() else "cpu")

# Training Setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.5f}")

# Train the model
train_model(model, train_loader)

# Testing function
def evaluate_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(actuals)

# Evaluate the model
y_pred, y_actual = evaluate_model(model, test_loader)

# Rescale predictions
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_actual_rescaled = scaler_y.inverse_transform(y_actual)

# Compute RMSE
rmse = np.sqrt(np.mean((y_pred_rescaled - y_actual_rescaled) ** 2))
print(f"Test RMSE: {rmse:.4f} °C")
