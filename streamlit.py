import os
import torch
import torch.nn as nn
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ------------------------------
# Configuration and Setup
# ------------------------------
st.set_page_config(page_title="EV Battery Temp Predictor", layout="centered")
st.title("🔋 Predict Internal Battery Temperature")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Battery dataset paths
battery_paths = {
    'LFP': 'battery/lfp_25degC',
    'NMC': 'battery/nmc_25degC',
    'NCA': 'battery/nca_25degC',
}

# Checkpoint paths
checkpoint_paths = {
    'LFP': 'checkpoints/lfp_model.pth',
    'NMC': 'checkpoints/nmc_model.pth',
    'NCA': 'checkpoints/nca_model.pth',
}

# ------------------------------
# Dataset + Model Definition
# ------------------------------
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

class TempLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ------------------------------
# Prediction Function
# ------------------------------
def evaluate_from_checkpoint(battery, external_temp):
    model = TempLSTM(input_size=5)
    model.load_state_dict(torch.load(checkpoint_paths[battery], map_location=device))
    model.to(device)
    model.eval()

    dataset = BatterySequenceDataset(battery_paths[battery], external_temp)
    dataloader = DataLoader(dataset, batch_size=64)

    preds = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            preds.extend(out.cpu().numpy())

    return sum(preds) / len(preds)

# ------------------------------
# Streamlit UI
# ------------------------------
external_temp = st.number_input(
    "🌡️ Enter External Temperature (°C):", 
    min_value=-20.0, 
    max_value=80.0, 
    value=25.0
)

battery_options = st.multiselect(
    "🔌 Select Battery Types:", 
    ["LFP", "NMC", "NCA"]
)

if st.button("🚀 Predict Internal Temp"):
    if not battery_options:
        st.warning("Please select at least one battery type.")
    else:
        with st.spinner("Evaluating models using saved checkpoints..."):
            results = {}
            for battery in battery_options:
                avg_temp = float(evaluate_from_checkpoint(battery, external_temp))
                results[battery] = avg_temp

            st.success("✅ Prediction complete!")

            # Display all predictions
            # for battery, temp in results.items():
            #     st.write(f"**{battery}** ➜ Predicted Internal Temp: `{temp:.2f}°C`")

            # Best battery suggestion
            best_battery = min(results, key=results.get)
            st.markdown(f"### 🏆 Best Battery at {external_temp}°C: **{best_battery}**")

            reasoning = {
                'LFP': "LFP batteries maintain better thermal stability under high temperatures, making them safer and more reliable.",
                'NMC': "NMC batteries strike a good balance between energy density and thermal control, performing well in moderate conditions.",
                'NCA': "NCA batteries have high energy density but tend to heat up more, which might affect safety in hot environments."
            }

            st.info(f"**Why {best_battery}?** {reasoning[best_battery]}")
