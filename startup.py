#!/usr/bin/env python3
"""
IoT Sentinel Startup Script
This script helps you start the IoT Sentinel system with proper checks and setup.
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                        IoT Sentinel                         ║
║                 AI-Driven Intrusion Detection               ║
║                      Startup Script                         ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher is required!")
        return False
    
    print("✅ Python version check passed")
    return True

def check_admin_privileges():
    """Check if running with admin/root privileges (needed for packet capture)"""
    system = platform.system().lower()
    
    if system == "windows":
        try:
            import ctypes
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
            if not is_admin:
                print("⚠️  WARNING: Not running as Administrator")
                print("   Packet capture may not work properly")
                print("   Consider running as Administrator")
                return False
        except:
            print("⚠️  Cannot determine admin status on Windows")
            return False
    else:  # Linux/Mac
        if os.geteuid() != 0:
            print("⚠️  WARNING: Not running as root")
            print("   Packet capture may not work properly") 
            print("   Consider running with sudo")
            return False
    
    print("✅ Admin privileges check passed")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'flask', 'flask_cors', 'scapy', 'pandas', 'sklearn', 
        'numpy', 'torch', 'scipy', 'psutil', 'waitress', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'server.py', 'config.toml', 'dashboard.html',
        'feature_extractor.py', 'ids_deep.py', 'packet_sniffer.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files are present")
    return True

def find_network_interfaces():
    """List available network interfaces"""
    try:
        from scapy.all import get_if_list
        interfaces = get_if_list()
        print(f"📡 Available network interfaces:")
        for i, iface in enumerate(interfaces, 1):
            print(f"   {i}. {iface}")
        return interfaces
    except Exception as e:
        print(f"❌ Error getting interfaces: {e}")
        return []

def create_directories():
    """Create necessary directories"""
    dirs = ['logs', 'models']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created/verified directory: {dir_name}")

def install_dependencies():
    """Install missing dependencies"""
    print("🔄 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def start_server():
    """Start the IoT Sentinel server"""
    print("\n🚀 Starting IoT Sentinel...")
    print("Dashboard will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Use the optimized server if available, otherwise fall back to original
        server_file = 'server.py'
        subprocess.run([sys.executable, server_file])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    print_banner()
    
    # Perform checks
    checks_passed = True
    
    print("🔍 Performing system checks...\n")
    
    if not check_python_version():
        checks_passed = False
    
    if not check_admin_privileges():
        print("   Continuing anyway (may affect packet capture)")
    
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n🔄 Attempting to install missing dependencies...")
        if not install_dependencies():
            checks_passed = False
    
    print("\n📁 Checking required files...")
    if not check_files():
        checks_passed = False
    
    if not checks_passed:
        print("\n❌ Some checks failed. Please resolve the issues above.")
        return 1
    
    print("\n📡 Network interface information:")
    find_network_interfaces()
    
    print("\n🔧 Setting up directories...")
    create_directories()
    
    print("\n✅ All checks passed!")
    
    # Ask user if they want to continue
    try:
        response = input("\nStart IoT Sentinel server? (y/n): ").lower().strip()
        if response in ['y', 'yes', '']:
            start_server()
        else:
            print("👋 Startup cancelled by user")
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)