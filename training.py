import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import re
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Dataset class
class BatterySequenceDataset(Dataset):
    def __init__(self, folder_path, external_temp, seq_len=20):
        self.seq_len = seq_len
        self.inputs, self.targets = [], []
        self.scaler_X, self.scaler_y = MinMaxScaler(), MinMaxScaler()

        data = []
        for fname in os.listdir(folder_path):
            if fname.endswith('.xlsx'):
                df = pd.read_excel(os.path.join(folder_path, fname))
                if all(c in df.columns for c in ['Test_Time(s)', 'Voltage(V)', 'Current(A)', 'Surface_Temp(degC)']):
                    df = df[['Test_Time(s)', 'Voltage(V)', 'Current(A)', 'Surface_Temp(degC)']]
                    df['Charging_Current'] = self.extract_charging_current(fname)
                    df['Ext_Temp'] = external_temp
                    data.append(df.dropna())
        df_full = pd.concat(data, ignore_index=True)

        features = ['Test_Time(s)', 'Voltage(V)', 'Current(A)', 'Charging_Current', 'Ext_Temp']
        X = self.scaler_X.fit_transform(df_full[features])
        y = self.scaler_y.fit_transform(df_full[['Surface_Temp(degC)']])

        for i in range(len(X) - seq_len):
            self.inputs.append(torch.tensor(X[i:i+seq_len], dtype=torch.float32))
            self.targets.append(torch.tensor(y[i+seq_len], dtype=torch.float32))

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        return len(self.inputs)

    def extract_charging_current(self, fname):
        match = re.search(r'_(\d+\.?\d*)C', fname)
        return float(match.group(1)) if match else 0.0

# LSTM Model
class TempLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Training function
def train_model(model, dataloader, epochs=10, lr=1e-3):
    model.to(device)  # <-- Move model to GPU
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)  # <-- Move inputs to GPU
            y_batch = y_batch.to(device)  # <-- Move targets to GPU
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

# Define battery paths
battery_paths = {
    'LFP': 'battery/lfp_25degC',
    'NMC': 'battery/nmc_25degC',
    'NCA': 'battery/nca_25degC',
}

# Create checkpoints folder
os.makedirs("checkpoints", exist_ok=True)

# Train and save model for each battery
for name, path in battery_paths.items():
    print(f"\n🔧 Training model for {name} battery...")
    dataset = BatterySequenceDataset(path, external_temp=45.0)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = TempLSTM(input_size=5)
    train_model(model, dataloader)
    
    torch.save(model.state_dict(), f"checkpoints/{name.lower()}_model.pth")
    print(f"✅ Saved checkpoint: checkpoints/{name.lower()}_model.pth")
