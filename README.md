
# Federated-CT-FL

A complete federated learning system for CT-based body composition prediction. This project simulates multi-institutional training using Flower, PyTorch, and dummy medical imaging data.

## 🌟 Key Features
- Federated Learning using [Flower](https://flower.dev/)
- CNN-based regression for CT scan analysis
- Dummy data for standalone simulation
- Device-agnostic training (GPU/CPU)
- Modular code with logging, preprocessing, and metrics
- Optional Differential Privacy (DP)

## 📁 Project Structure

```
Federated-Learning-Framework-for-Biomedical-Image-Regression-Tasks/
├── data/               # Placeholder for client datasets
├── src/                # Model, training, evaluation code
├── fl_simulation/      # Federated server/client logic
├── utils/              # Logging, data splitting, DP
├── configs/            # Config files and (legacy) SLURM script
├── notebooks/          # Optional Jupyter analysis
├── results/            # Logs and model checkpoints
├── run_local_fl.sh     # Local FL runner (multi-client simulation)
├── requirements.txt    # Python dependencies
└── README.md           # You are here
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/https://github.com/noumannahmad/Federated-Learning-Framework-for-Biomedical-Image-Regression-Tasks.git
cd Federated-CT-FL
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Federated Simulation Locally
```bash
chmod +x run_local_fl.sh
./run_local_fl.sh
```

### 4. Run Individual Scripts (Optional)
```bash
python fl_simulation/server.py        # Start FL server
python fl_simulation/client.py        # Start FL client
```

## 📊 Outputs
- Model checkpoints: `results/models`
- Training logs: `results/logs`
- Sample performance metrics: MAE, R²

## 📚 Future Additions
- Use real medical imaging datasets (.npy, .nii.gz)
- Add client authentication and secure aggregation
- Deploy on Kubernetes or GCP with Docker

## 📄 License
MIT License © 2025 Mansoor Hayat
