import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

DEVICE = torch.device("cpu")

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(9, 6), nn.ReLU(),
            nn.Linear(6, 3), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(3, 6), nn.ReLU(),
            nn.Linear(6, 9)
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

class IDSEngine:
    def __init__(self):
        self.model = AutoEncoder().to(DEVICE)
        self.scaler = StandardScaler()
        self.model_path = "models/ae_model.pt"
        self.scaler_path = "models/ae_scaler.pkl"
        self.threshold_path = "models/threshold.pkl"
        self.threshold = 0.0
        self.feature_names = [
            'protocol', 'packet_length', 'ttl', 'tcp_flags',
            'window_size', 'payload_length', 'entropy',
            'payload_ratio', 'dst_port'
        ]

    def train_model(self, contamination=0.1, epochs=30, lr=1e-3):
        np.random.seed(42)
        normal = pd.DataFrame({
            'protocol': np.random.choice([0, 1], 2000, p=[.7, .3]),
            'packet_length': np.random.normal(500, 100, 2000).astype(int),
            'ttl': np.random.normal(64, 10, 2000).astype(int),
            'tcp_flags': np.random.choice([16, 24], 2000, p=[.8, .2]),
            'window_size': np.random.normal(64240, 1000, 2000).astype(int),
            'payload_length': np.random.normal(200, 50, 2000).astype(int),
            'entropy': np.random.normal(4, 1, 2000).clip(0, 8),
            'payload_ratio': np.random.normal(0.4, 0.1, 2000).clip(0, 1),
            'dst_port': np.random.choice([80, 443, 22, 23, 1883], 2000)
        })
        self.scaler.fit(normal)
        normal_scaled = self.scaler.transform(normal)
        tensor = torch.tensor(normal_scaled, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon = self.model(tensor)
            loss = criterion(recon, tensor)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                print(f"Epoch {epoch:02d} | loss = {loss.item():.4f}")

        with torch.no_grad():
            val_recon = self.model(tensor)
            val_error = torch.mean((val_recon - tensor)**2, dim=1)
            self.threshold = np.percentile(val_error.numpy(), 95)
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.threshold, self.threshold_path)
        print(f"Training done – threshold = {self.threshold:.4f}")
        return self.model

    def load_model(self):
        if all(os.path.exists(f) for f in [self.model_path, self.scaler_path, self.threshold_path]):
            self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
            self.scaler = joblib.load(self.scaler_path)
            self.threshold = joblib.load(self.threshold_path)
            return True
        return False

    def predict(self, df):
        expected_columns = self.feature_names
        # This function is great practice as it ensures column order
        df = ensure_dataframe_structure(df, expected_columns) 
        
        X = self.scaler.transform(df) # Pass the DataFrame directly
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            mse = torch.mean((recon - X_tensor)**2, dim=1).numpy()
        return (mse > self.threshold).astype(int)

    def evaluate(self, X_test, y_test):
        preds = self.predict(pd.DataFrame(X_test, columns=self.feature_names))
        print(classification_report(y_test, preds, target_names=['Normal', 'Threat']))
        print("ROC-AUC:", roc_auc_score(y_test, preds))
        return preds

def ensure_dataframe_structure(df, expected_columns):
    # Check if all expected columns are present
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    # Reorder columns to match the expected structure
    df = df[expected_columns]
    return df