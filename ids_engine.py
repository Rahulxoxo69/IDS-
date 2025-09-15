import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

class IDSEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = "models/ids_model.pkl"
        self.scaler_path = "models/scaler.pkl"
        
    def train_model(self, contamination=0.1):
        """Train the anomaly detection model"""
        # Create a synthetic dataset for training
        # In a real scenario, you would use actual network traffic data
        np.random.seed(42)
        
        # Normal traffic patterns
        normal_protocol = np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # Mostly TCP, some UDP
        normal_length = np.random.normal(500, 100, 1000).astype(int)
        normal_ttl = np.random.normal(64, 10, 1000).astype(int)
        normal_flags = np.random.choice([16, 24], 1000, p=[0.8, 0.2])  # Mostly ACK, some ACK+PSH
        normal_window = np.random.normal(64240, 1000, 1000).astype(int)
        normal_payload = np.random.normal(200, 50, 1000).astype(int)
        
        normal_data = np.column_stack((
            normal_protocol, normal_length, normal_ttl, 
            normal_flags, normal_window, normal_payload
        ))
        
        # Anomalous traffic patterns (various attack patterns)
        anomalous_data = []
        
        # Port scan pattern (many SYN packets)
        for _ in range(50):
            anomalous_data.append([0, 60, 64, 2, 1024, 0])  # SYN packet
        
        # DDoS pattern (many large packets)
        for _ in range(50):
            anomalous_data.append([0, 1500, 255, 16, 65535, 1460])  # Large ACK packet
            
        # Malformed packets
        for _ in range(50):
            anomalous_data.append([3, 40, 1, 0, 0, 0])  # Unknown protocol, low TTL
            
        anomalous_data = np.array(anomalous_data)
        
        # Combine normal and anomalous data
        X = np.vstack((normal_data, anomalous_data))
        y = np.hstack((np.zeros(len(normal_data)), np.ones(len(anomalous_data))))

        # Ensure X is a numpy array (no feature names)
        X = np.array(X)
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest model
        self.model = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_estimators=100
        )
        self.model.fit(X_scaled)
        
        # Save model and scaler
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        return self.model
    
    def load_model(self):
        """Load a pre-trained model"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False
    
    def predict(self, features_df):
    
        if self.model is None:
            raise Exception("Model not trained or loaded")
        
        if features_df.empty:
            return np.array([])
        
        # Convert to numpy array (removes feature names)
        X = features_df.values
        X = np.array(X)
        # Scale the features
        X_scaled = self.scaler.transform(X)  # Now no feature name warning
    
    # Predict anomalies (-1 for anomalies, 1 for normal)
        predictions = self.model.predict(X_scaled)
    
    # Convert to binary (0 for normal, 1 for anomaly)
        return (predictions == -1).astype(int)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise Exception("Model not trained or loaded")
            
        X_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_scaled)
        binary_predictions = (predictions == -1).astype(int)
        
        accuracy = np.mean(binary_predictions == y_test)
        return accuracy