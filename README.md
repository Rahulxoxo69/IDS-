IoT Sentinel: AI-Powered Network Intrusion Detection System
https://img.shields.io/badge/IoT-Sentinel-blue
https://img.shields.io/badge/Python-3.8%252B-green
https://img.shields.io/badge/ML-Autoencoder-orange
https://img.shields.io/badge/UI-Realtime%2520Dashboard-purple
https://img.shields.io/badge/License-MIT-yellow

IoT Sentinel is a comprehensive AI-driven intrusion detection system designed specifically for monitoring and securing IoT networks. It combines deep learning anomaly detection with real-time packet analysis and a beautiful dashboard interface.

🌟 Features
Real-time Packet Analysis: Captures and analyzes network traffic in real-time

Deep Learning Detection: Uses PyTorch-based autoencoder for anomaly detection

Beautiful Dashboard: Modern, responsive web interface with live metrics

Multi-Platform: Works on Windows, Linux, and Raspberry Pi

Easy Deployment: Comprehensive setup scripts and documentation

Raspberry Pi Optimized: Special configurations for low-power devices

🏗️ Architecture
text
Network Traffic → Packet Capture → Feature Extraction → AI Analysis → Dashboard Alert
      ↑               ↓                ↓                 ↓              ↓
   IoT Devices   Packet Sniffer   Feature Extractor   IDS Engine   Web Interface
📋 Requirements
Hardware Requirements
Minimum: 2GB RAM, Dual-core CPU, 10GB storage

Recommended: 4GB+ RAM, Quad-core CPU, 50GB storage

Raspberry Pi: Raspberry Pi 4 (2GB+ RAM) with heatsinks

Software Requirements
Python 3.8 or higher

Network interface with promiscuous mode support

Administrative/root privileges for packet capture

📦 Installation
Option 1: Automated Setup (Recommended)
bash
# Clone the repository
git clone https://github.com/your-username/iot-sentinel.git
cd iot-sentinel

# Run the setup script
python startup.py
Option 2: Manual Installation
bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize the system
python server.py
🔧 Configuration
Edit config.toml to customize your setup:

toml
[server]
interface = "eth0"           # Network interface to monitor
flask_host = "0.0.0.0"       # Dashboard host
flask_port = 5000            # Dashboard port
max_alerts = 100             # Maximum alerts to store
batch_size = 10              # Packets to process per batch

[model]
contamination = 0.1          # Expected anomaly rate
epochs = 30                  # Training epochs
learning_rate = 0.001        # Learning rate

[auth]
hmac_key = "your-secret-key" # Change this in production!
🚀 Usage
Starting the System
bash
# Start the IDS server
python server.py

# Access the dashboard at http://localhost:5000
Training the Model
bash
# Train with default parameters
python -c "from ids_deep import IDSEngine; engine = IDSEngine(); engine.train_model()"

# Train with custom dataset
python train_with_kaggle_data.py
Monitoring Network Traffic
bash
# Test packet capture
python test_packet_capture.py

# Generate test traffic
python net_gen.py --duration 60
🍓 Raspberry Pi Deployment
Prerequisites
Raspberry Pi 4 with Raspberry Pi OS (64-bit)

Ethernet connection for network monitoring

Setup
bash
# Run the Pi-optimized setup
chmod +x deploy_pi.sh
./deploy_pi.sh

# Or manually configure
python pi_optimize.py
Pi-specific Features
Reduced memory footprint

Temperature monitoring

Optimized model architecture

System service auto-start

📊 Dashboard Features
The web dashboard provides:

Real-time traffic monitoring

Threat level visualization

Device distribution charts

Alert history and details

System health metrics

Raspberry Pi status monitoring

🔍 Diagnostics
bash
# Comprehensive system check
python diag.py

# Interface discovery
python find_interface.py

# Performance testing
python test_packet_capture.py
🗃️ Data Management
Using Kaggle Datasets
Download a network traffic dataset from Kaggle

Preprocess it using preprocess_kaggle_data.py

Train the model: python train_with_kaggle_data.py

Custom Datasets
Ensure your dataset includes these features:

Protocol type

Packet length

TTL values

TCP flags

Port numbers

Payload characteristics

🛠️ Troubleshooting
Common Issues
Packet Capture Permissions

bash
# Linux/Mac
sudo python server.py

# Windows: Run as Administrator
Missing Dependencies

bash
pip install -r requirements.txt --force-reinstall
Interface Not Found

bash
python find_interface.py
# Update config.toml with the correct interface
Model Training Issues

bash
python refresh.py  # Clears old model files
python server.py   # Retrains automatically
📈 Performance Tips
For Raspberry Pi:

Use wired Ethernet connection

Add heatsinks for continuous operation

Use config_pi.toml for optimized settings

For High-Traffic Networks:

Increase batch_size in config

Use a more powerful machine

Consider port mirroring on your switch

For Better Detection:

Train with real attack datasets

Regularly update the model

Fine-tune detection thresholds

🤝 Contributing
Fork the repository

Create a feature branch: git checkout -b feature-name

Commit changes: git commit -am 'Add feature'

Push to branch: git push origin feature-name

Submit a pull request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Scapy community for packet manipulation library

PyTorch team for deep learning framework

Raspberry Pi Foundation for hardware support

Cybersecurity researchers for attack datasets

📞 Support
For issues and questions:

Check the troubleshooting section

Review open GitHub issues

Create a new issue with detailed information

Note: This system is designed for educational and research purposes. Always ensure you have proper authorization before monitoring any network.
