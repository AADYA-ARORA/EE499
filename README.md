# 🔋 EV Battery Temperature Predictor

An advanced deep learning system that predicts internal battery temperatures for different types of EV batteries (LFP, NMC, NCA) using LSTM neural networks. The system helps in selecting the optimal battery type based on external temperature conditions.

![Battery Temperature Prediction](assets/prediction_demo.png)

## 🌟 Features

- Real-time temperature prediction for three battery types:
  - Lithium Iron Phosphate (LFP)
  - Nickel Manganese Cobalt (NMC)
  - Nickel Cobalt Aluminum (NCA)
- Interactive Streamlit web interface
- LSTM-based sequence modeling
- Automated battery type recommendations
- Cross-chemistry transfer learning capabilities

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ev-battery-temp-predictor.git
cd ev-battery-temp-predictor
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Project Structure

```
ev-battery-temp-predictor/
├── battery/                  # Battery data files
│   ├── lfp_25degC/          # LFP battery data
│   ├── nmc_25degC/          # NMC battery data
│   └── nca_25degC/          # NCA battery data
├── checkpoints/             # Trained model checkpoints
├── streamlit.py            # Streamlit web application
├── training.py             # Model training script
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## 🚀 Usage

### Training Models

To train models for all battery types:

```bash
python training.py
```

This will:
- Process the battery data
- Train LSTM models for each battery type
- Save model checkpoints in the `checkpoints/` directory

### Running the Web Interface

Launch the Streamlit application:

```bash
streamlit run streamlit.py
```

Navigate to `http://localhost:8501` in your web browser to:
- Input external temperature conditions
- Select battery types for comparison
- Get temperature predictions and recommendations

## 🧪 Model Architecture

The system uses a Long Short-Term Memory (LSTM) neural network with:
- Input features: Time, Voltage, Current, Charging Rate, External Temperature
- 2 LSTM layers with 64 hidden units
- Sequence length of 20 timesteps
- MinMax scaling for feature normalization

## 📈 Performance

Cross-validation results for different battery combinations:

| Training Data | Test Data | MAE (°C) | R² Score |
|--------------|-----------|----------|----------|
| LFP+NMC      | NCA       | 1.32     | 0.25     |
| LFP+NCA      | NMC       | 1.92     | -1.51    |
| NMC+NCA      | LFP       | 1.63     | -0.28    |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a fork Request.

## 📧 Contact

For questions or feedback, please reach out to [aadya.arora@iitgn.ac.in](mailto:aadya.arora@iitgn.ac.in) or [siya.patil@iitgn.ac.in](mailto:siya.patil@iitgn.ac.in)

---
Made with ❤️ for safer and more efficient EV batteries
