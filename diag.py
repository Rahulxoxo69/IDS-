#!/usr/bin/env python3
"""
IoT Sentinel Diagnostic Script
This script will help identify why packets aren't being captured/displayed
"""

import sys
import time
import json
import requests
from datetime import datetime

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def test_scapy_import():
    """Test if Scapy can be imported and basic functionality works"""
    print_header("1. SCAPY IMPORT TEST")
    
    try:
        from scapy.all import get_if_list, sniff, IP
        print("✅ Scapy imported successfully")
        
        # Test interface listing
        interfaces = get_if_list()
        print(f"✅ Found {len(interfaces)} network interfaces:")
        for i, iface in enumerate(interfaces):
            print(f"   {i+1}. {iface}")
        return interfaces
        
    except ImportError as e:
        print(f"❌ Failed to import Scapy: {e}")
        print("   Install with: pip install scapy")
        return None
    except Exception as e:
        print(f"❌ Scapy error: {e}")
        return None

def test_packet_capture(interfaces):
    """Test packet capture on each interface"""
    if not interfaces:
        return False
    
    print_header("2. PACKET CAPTURE TEST")
    
    from scapy.all import sniff, IP
    
    success_count = 0
    
    for iface in interfaces:
        # Skip certain interfaces that are unlikely to work
        if any(skip in iface.lower() for skip in ['loopback', 'bluetooth', 'vmware', 'virtualbox']):
            print(f"⏭️  Skipping {iface} (likely virtual/loopback)")
            continue
            
        print(f"\n🔍 Testing interface: {iface}")
        
        try:
            print("   Attempting to capture 3 packets (5-second timeout)...")
            packets = sniff(iface=iface, count=3, timeout=5, quiet=True)
            
            if len(packets) > 0:
                print(f"   ✅ Captured {len(packets)} packets!")
                
                # Check for IP packets
                ip_packets = [p for p in packets if p.haslayer(IP)]
                print(f"   📦 IP packets: {len(ip_packets)}")
                
                if ip_packets:
                    print("   📋 Sample packet info:")
                    pkt = ip_packets[0]
                    print(f"      Source: {pkt[IP].src}")
                    print(f"      Destination: {pkt[IP].dst}")
                    print(f"      Protocol: {pkt[IP].proto}")
                    print(f"      Length: {len(pkt)} bytes")
                
                success_count += 1
                
                # Ask if user wants to use this interface
                try:
                    choice = input(f"\n   Use {iface} for IoT Sentinel? (y/n/skip): ").lower()
                    if choice in ['y', 'yes']:
                        return iface
                    elif choice in ['skip', 's']:
                        break
                except KeyboardInterrupt:
                    return None
                    
            else:
                print("   ⚠️  No packets captured (interface may be inactive)")
                
        except PermissionError:
            print("   ❌ Permission denied - try running as administrator/root")
        except OSError as e:
            print(f"   ❌ OS Error: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if success_count == 0:
        print("\n❌ No interfaces successfully captured packets!")
        print("   This might be due to:")
        print("   - Insufficient permissions (try sudo/administrator)")
        print("   - No active network traffic")
        print("   - Firewall blocking packet capture")
        return False
    else:
        print(f"\n✅ {success_count} interface(s) can capture packets")
        return True

def generate_test_traffic():
    """Generate some network traffic for testing"""
    print_header("3. GENERATING TEST TRAFFIC")
    
    print("🌐 Generating test network traffic...")
    
    try:
        import requests
        test_urls = [
            'http://httpbin.org/json',
            'http://httpbin.org/ip',
            'http://httpbin.org/user-agent'
        ]
        
        for url in test_urls:
            try:
                print(f"   📡 Requesting: {url}")
                response = requests.get(url, timeout=5)
                print(f"   ✅ Response: {response.status_code}")
                time.sleep(1)
            except Exception as e:
                print(f"   ⚠️  Failed: {e}")
                
    except ImportError:
        print("   ⚠️  requests not available, trying basic socket connection")
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('8.8.8.8', 53))
            sock.close()
            if result == 0:
                print("   ✅ Basic network connectivity confirmed")
            else:
                print("   ⚠️  Network connectivity issues")
        except Exception as e:
            print(f"   ❌ Network test failed: {e}")

def test_api_connection():
    """Test connection to IoT Sentinel API"""
    print_header("4. API CONNECTION TEST")
    
    api_urls = [
        'http://127.0.0.1:5000/stats',
        'http://localhost:5000/stats',
        'http://127.0.0.1:5000/health',
        'http://127.0.0.1:5000/debug'
    ]
    
    for url in api_urls:
        try:
            print(f"🔗 Testing: {url}")
            response = requests.get(url, timeout=3)
            
            if response.status_code == 200:
                print(f"   ✅ Success! Status: {response.status_code}")
                
                try:
                    data = response.json()
                    print(f"   📊 Data preview:")
                    print(f"      Total packets: {data.get('total_packets', 'N/A')}")
                    print(f"      Total threats: {data.get('total_threats', 'N/A')}")
                    print(f"      Interface: {data.get('interface_info', {}).get('selected', 'N/A')}")
                    
                    if 'debug' in url and 'recent_packet_count' in data:
                        print(f"      Recent packets in buffer: {data['recent_packet_count']}")
                        
                except:
                    print(f"   📄 Raw response: {response.text[:200]}...")
                    
                return True
            else:
                print(f"   ❌ Failed with status: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection refused (server not running?)")
        except requests.exceptions.Timeout:
            print(f"   ❌ Timeout")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return False

def test_feature_extraction():
    """Test if feature extraction is working"""
    print_header("5. FEATURE EXTRACTION TEST")
    
    try:
        from scapy.all import IP, TCP
        from feature_extractor import FeatureExtractor
        
        # Create a test packet
        test_packet = IP(src="192.168.1.100", dst="192.168.1.1")/TCP(sport=12345, dport=80)
        
        extractor = FeatureExtractor()
        print("✅ FeatureExtractor created")
        
        # Test feature extraction
        features = extractor.extract_features([test_packet])
        print("✅ Feature extraction successful")
        print(f"   Features shape: {features.shape}")
        print(f"   Feature columns: {list(features.columns)}")
        print(f"   Sample values: {features.iloc[0].tolist()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return False

def monitor_live_capture():
    """Monitor live packet capture for a few seconds"""
    print_header("6. LIVE CAPTURE MONITOR")
    
    try:
        from scapy.all import sniff, IP
        
        print("🔴 Starting 10-second live capture monitor...")
        print("   (Browse the web or generate traffic to see packets)")
        print("   Press Ctrl+C to stop early")
        
        packet_count = 0
        ip_count = 0
        start_time = time.time()
        
        def packet_callback(packet):
            nonlocal packet_count, ip_count
            packet_count += 1
            if packet.haslayer(IP):
                ip_count += 1
                print(f"   📦 Packet {packet_count}: {packet[IP].src} → {packet[IP].dst} ({len(packet)} bytes)")
        
        try:
            sniff(filter="ip", prn=packet_callback, timeout=10, store=False)
        except KeyboardInterrupt:
            pass
        
        duration = time.time() - start_time
        print(f"\n📊 Capture summary:")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Total packets: {packet_count}")
        print(f"   IP packets: {ip_count}")
        print(f"   Rate: {packet_count/duration:.1f} packets/second")
        
        if packet_count == 0:
            print("\n⚠️  WARNING: No packets captured!")
            print("   Possible causes:")
            print("   - No network activity")
            print("   - Wrong interface selected")
            print("   - Permission issues")
            print("   - Firewall blocking capture")
        
        return packet_count > 0
        
    except Exception as e:
        print(f"❌ Live capture error: {e}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║             IoT Sentinel Diagnostic Tool                    ║
║         Let's find out why packets aren't showing!          ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    print("This diagnostic will help identify packet capture issues.")
    print("Please ensure IoT Sentinel server is running in another terminal.")
    
    try:
        input("\nPress Enter to start diagnostics (or Ctrl+C to exit)...")
    except KeyboardInterrupt:
        print("\n👋 Diagnostics cancelled")
        return
    
    # Run diagnostic tests
    interfaces = test_scapy_import()
    
    if interfaces:
        working_interface = test_packet_capture(interfaces)
        
        if working_interface:
            print(f"\n✅ Recommended interface: {working_interface}")
            print("   Update your config.toml with this interface")
    
    generate_test_traffic()
    
    server_running = test_api_connection()
    
    if not server_running:
        print("\n❌ Server not responding! Start it with:")
        print("   python server.py")
        return
    
    test_feature_extraction()
    
    # Final live test
    print("\n" + "="*60)
    print("  FINAL TEST: Live packet monitoring")
    print("="*60)
    
    monitor_live_capture()
    
    # Recommendations
    print_header("RECOMMENDATIONS")
    
    if not interfaces:
        print("❌ Install Scapy: pip install scapy")
    elif not working_interface:
        print("❌ Run as administrator/root for packet capture")
    elif not server_running:
        print("❌ Start IoT Sentinel server: python server.py")
    else:
        print("✅ System appears to be working!")
        print("   If dashboard still shows no packets:")
        print("   1. Generate network traffic (browse web, download files)")
        print("   2. Check browser console for JavaScript errors")
        print("   3. Try refreshing the dashboard")
        print("   4. Check the server logs in logs/ids_server.log")

if __name__ == "__main__":
    main()