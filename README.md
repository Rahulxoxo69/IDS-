
```markdown
# IoT Sentinel: AI-Powered Network Intrusion Detection System

![IoT Sentinel](https://img.shields.io/badge/IoT-Sentinel-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Machine Learning](https://img.shields.io/badge/ML-Autoencoder-orange)
![Dashboard](https://img.shields.io/badge/UI-Realtime%20Dashboard-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

IoT Sentinel is a comprehensive AI-driven intrusion detection system designed specifically for monitoring and securing IoT networks. It combines deep learning anomaly detection with real-time packet analysis and a beautiful dashboard interface.

## 🌟 Features

- **Real-time Packet Analysis**: Captures and analyzes network traffic in real-time
- **Deep Learning Detection**: Uses PyTorch-based autoencoder for anomaly detection
- **Beautiful Dashboard**: Modern, responsive web interface with live metrics
- **Multi-Platform**: Works on Windows, Linux, and Raspberry Pi
- **Easy Deployment**: Comprehensive setup scripts and documentation
- **Raspberry Pi Optimized**: Special configurations for low-power devices

## 🏗️ Architecture

```
Network Traffic → Packet Capture → Feature Extraction → AI Analysis → Dashboard Alert
      ↑               ↓                ↓                 ↓              ↓
   IoT Devices   Packet Sniffer   Feature Extractor   IDS Engine   Web Interface
```

## 📋 Requirements

### Hardware Requirements
- **Minimum**: 2GB RAM, Dual-core CPU, 10GB storage
- **Recommended**: 4GB+ RAM, Quad-core CPU, 50GB storage
- **Raspberry Pi**: Raspberry Pi 4 (2GB+ RAM) with heatsinks

### Software Requirements
- Python 3.8 or higher
- Network interface with promiscuous mode support
- Administrative/root privileges for packet capture

## 📦 Installation

### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/iot-sentinel.git
cd iot-sentinel

# Run the setup script
python startup.py
```

### Option 2: Manual Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize the system
python server.py
```

## 🔧 Configuration

Edit `config.toml` to customize your setup:

```toml
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
```

## 🚀 Usage

### Starting the System
```bash
# Start the IDS server
python server.py

# Access the dashboard at http://localhost:5000
```

### Training the Model
```bash
# Train with default parameters
python -c "from ids_deep import IDSEngine; engine = IDSEngine(); engine.train_model()"

# Train with custom dataset
python train_with_kaggle_data.py
```

### Monitoring Network Traffic
```bash
# Test packet capture
python test_packet_capture.py

# Generate test traffic
python net_gen.py --duration 60
```

## 🍓 Raspberry Pi Deployment

### Prerequisites
- Raspberry Pi 4 with Raspberry Pi OS (64-bit)
- Ethernet connection for network monitoring

### Setup
```bash
# Run the Pi-optimized setup
chmod +x deploy_pi.sh
./deploy_pi.sh

# Or manually configure
python pi_optimize.py
```

### Pi-specific Features
- Reduced memory footprint
- Temperature monitoring
- Optimized model architecture
- System service auto-start

## 📊 Dashboard Features

The web dashboard provides:
- Real-time traffic monitoring
- Threat level visualization
- Device distribution charts
- Alert history and details
- System health metrics
- Raspberry Pi status monitoring

## 🔍 Diagnostics

```bash
# Comprehensive system check
python diag.py

# Interface discovery
python find_interface.py

# Performance testing
python test_packet_capture.py
```

## 🗃️ Data Management

### Using Kaggle Datasets
1. Download a network traffic dataset from Kaggle
2. Preprocess it using `preprocess_kaggle_data.py`
3. Train the model: `python train_with_kaggle_data.py`

### Custom Datasets
Ensure your dataset includes these features:
- Protocol type
- Packet length
- TTL values
- TCP flags
- Port numbers
- Payload characteristics

## 🛠️ Troubleshooting

### Common Issues

1. **Packet Capture Permissions**
   ```bash
   # Linux/Mac
   sudo python server.py
   
   # Windows: Run as Administrator
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

3. **Interface Not Found**
   ```bash
   python find_interface.py
   # Update config.toml with the correct interface
   ```

4. **Model Training Issues**
   ```bash
   python refresh.py  # Clears old model files
   python server.py   # Retrains automatically
   ```

## 📈 Performance Tips

1. **For Raspberry Pi**:
   - Use wired Ethernet connection
   - Add heatsinks for continuous operation
   - Use `config_pi.toml` for optimized settings

2. **For High-Traffic Networks**:
   - Increase `batch_size` in config
   - Use a more powerful machine
   - Consider port mirroring on your switch

3. **For Better Detection**:
   - Train with real attack datasets
   - Regularly update the model
   - Fine-tune detection thresholds

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Scapy community for packet manipulation library
- PyTorch team for deep learning framework
- Raspberry Pi Foundation for hardware support
- Cybersecurity researchers for attack datasets

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review open GitHub issues
3. Create a new issue with detailed information

---

**Note**: This system is designed for educational and research purposes. Always ensure you have proper authorization before monitoring any network.

---

# Requirements

```txt
# requirements.txt
# IoT Sentinel - AI-Powered Network Intrusion Detection System

# Core Framework
flask==2.3.3
flask-cors==4.0.0
waitress==2.1.2

# Packet Processing
scapy==2.5.0

# Data Processing & ML
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.10.1
joblib==1.3.2

# Deep Learning
torch==1.13.1
torchvision==0.14.1

# System Monitoring
psutil==5.9.5

# Configuration
tomli==2.0.1

# Networking & Security
requests==2.31.0
paramiko==3.2.0

# Utilities
tqdm==4.65.0
python-dateutil==2.8.2
pytz==2023.3

# Raspberry Pi Specific (optional)
RPi.GPIO==0.7.1
gpiozero==1.6.2

# Development & Testing
pytest==7.4.0
pytest-cov==4.1.0
black==23.7.0
flake8==6.0.0
```

To install these dependencies:

```bash
pip install -r requirements.txt
```

For Raspberry Pi, use the Pi-optimized installation:

```bash
pip install -r requirements.txt --no-deps
pip install torch==1.10.0 torchvision==0.11.0 -f https://torch.kmtea.eu/whl/stable.html
pip install scapy==2.5.0
```
```

To save this as a README.md file:

1. Copy all the text above
2. Create a new file in your text editor
3. Paste the content
4. Save the file as "README.md"

If you're looking for the actual project files to download, you would need to create them based on the code snippets provided in our previous conversation, or I can help you create any specific file you need.
