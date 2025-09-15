import scapy.all as scapy
import time

def test_packet_capture():
    print("Testing packet capture...")
    
    # Get available interfaces
    interfaces = scapy.get_if_list()
    print("Available interfaces:", interfaces)
    
    # Try to capture on each interface
    for iface in interfaces:
        if "Loopback" not in iface:  # Skip loopback
            print(f"\nTrying interface: {iface}")
            try:
                # Capture for 5 seconds
                packets = scapy.sniff(iface=iface, timeout=5, count=10)
                print(f"Captured {len(packets)} packets on {iface}")
                if packets:
                    for i, pkt in enumerate(packets[:3]):  # Show first 3 packets
                        print(f"  Packet {i+1}: {pkt.summary()}")
            except Exception as e:
                print(f"Error on {iface}: {e}")

if __name__ == "__main__":
    test_packet_capture()