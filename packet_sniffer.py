from scapy.all import sniff, IP, get_if_list
import logging
import time
import platform
import re

logger = logging.getLogger("PacketSniffer")

class PacketSniffer:
    def __init__(self, interface=None):
        self.interface = interface
        self.last_capture_time = 0
        self.capture_stats = {
            "total_captured": 0,
            "total_filtered": 0,
            "last_batch_size": 0,
            "last_capture_success": True
        }
        
        if not self.interface:
            self.interface = self._find_best_interface()
        else:
            # Test the provided interface
            if not self.test_interface(self.interface):
                logger.warning(f"Configured interface {self.interface} may not work. Trying to find a better one.")
                self.interface = self._find_best_interface()
        
        logger.info(f"PacketSniffer initialized with interface: {self.interface}")
    
    def _find_best_interface(self):
        """Find the best available network interface"""
        interfaces = get_if_list()
        
        # For Windows, look for interface names that contain keywords
        if platform.system() == "Windows":
            # Common Windows interface patterns
            windows_patterns = [
                r'Wi-?Fi',
                r'Ethernet',
                r'Local Area Connection',
                r'Wireless'
            ]
            
            for pattern in windows_patterns:
                for iface in interfaces:
                    if re.search(pattern, iface, re.IGNORECASE):
                        logger.info(f"Selected Windows interface: {iface}")
                        return iface
        
        # For other OS or fallback
        priority = ['eth0', 'wlan0', 'en0', 'Wi-Fi', 'Ethernet']
        
        for iface_name in priority:
            for iface in interfaces:
                if iface_name.lower() in iface.lower():
                    return iface
        
        # Find first non-loopback interface
        for iface in interfaces:
            if 'loopback' not in iface.lower() and iface.lower() not in ['lo', 'lo0']:
                return iface
        
        # Last resort - use any interface
        return interfaces[0] if interfaces else None
    
    def sniff_batch(self, count=10, timeout=1):
        """
        Capture a batch of packets with improved error handling and performance
        """
        try:
            start_time = time.time()
            
            # Use optimized filter for better performance
            packets = sniff(
                iface=self.interface,
                filter="ip",  # Only capture IP packets
                count=count,
                timeout=timeout,
                quiet=True,
                store=True
            )
            
            capture_time = time.time() - start_time
            self.last_capture_time = capture_time
            
            # Update statistics
            self.capture_stats["total_captured"] += len(packets)
            self.capture_stats["last_batch_size"] = len(packets)
            self.capture_stats["last_capture_success"] = True
            
            # Additional filtering for IP packets (double-check)
            ip_packets = [pkt for pkt in packets if pkt.haslayer(IP)]
            self.capture_stats["total_filtered"] += len(ip_packets)
            
            if len(packets) > 0:
                logger.info(f"Captured {len(packets)} packets ({len(ip_packets)} IP) in {capture_time:.3f}s on {self.interface}")
            else:
                logger.debug(f"No packets captured in {timeout}s on {self.interface}")
            
            return ip_packets
            
        except PermissionError:
            logger.error(f"Permission denied accessing interface {self.interface}. Try running as administrator.")
            self.capture_stats["last_capture_success"] = False
            return []
        except OSError as e:
            if "No such device" in str(e):
                logger.error(f"Interface {self.interface} not found. Available: {get_if_list()}")
                # Try to switch to a different interface
                self.interface = self._find_best_interface()
                logger.info(f"Switched to interface: {self.interface}")
                self.capture_stats["last_capture_success"] = False
            else:
                logger.error(f"OS Error capturing packets: {e}")
                self.capture_stats["last_capture_success"] = False
            return []
        except Exception as e:
            logger.error(f"Unexpected error capturing packets: {e}")
            self.capture_stats["last_capture_success"] = False
            return []
    
    def get_stats(self):
        """Get capture statistics"""
        return {
            "interface": self.interface,
            "available_interfaces": get_if_list(),
            "capture_stats": self.capture_stats.copy(),
            "last_capture_time": self.last_capture_time
        }
    
    def test_interface(self, interface=None):
        """Test if an interface can capture packets"""
        test_iface = interface or self.interface
        try:
            test_packets = sniff(iface=test_iface, count=1, timeout=2, quiet=True)
            return len(test_packets) > 0
        except:
            return False
    
    def switch_interface(self, new_interface):
        """Switch to a different network interface"""
        if self.test_interface(new_interface):
            old_interface = self.interface
            self.interface = new_interface
            logger.info(f"Switched interface from {old_interface} to {new_interface}")
            return True
        else:
            logger.warning(f"Cannot switch to interface {new_interface} - not available or no permission")
            return False