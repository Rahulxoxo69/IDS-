# Delete the old model files to force retraining
import os
if os.path.exists("models/ids_model.pkl"):
    os.remove("models/ids_model.pkl")
if os.path.exists("models/scaler.pkl"):
    os.remove("models/scaler.pkl")

# Then restart your server - it will retrain automatically
