#!/usr/bin/env python3
"""
Network Traffic Generator for IoT Sentinel Testing
Generates various types of network traffic to test packet capture
"""

import threading
import time
import socket
import random
import requests
import sys
from concurrent.futures import ThreadPoolExecutor

class TrafficGenerator:
    def __init__(self):
        self.running = False
        self.stats = {
            'http_requests': 0,
            'tcp_connections': 0,
            'udp_packets': 0,
            'dns_queries': 0
        }

    def generate_http_traffic(self):
        """Generate HTTP traffic"""
        urls = [
            'http://httpbin.org/json',
            'http://httpbin.org/ip',
            'http://httpbin.org/headers',
            'http://httpbin.org/user-agent',
            'http://httpbin.org/delay/1',
            'https://api.github.com/users/octocat',
            'https://jsonplaceholder.typicode.com/posts/1',
            'https://jsonplaceholder.typicode.com/users',
        ]
        
        while self.running:
            try:
                url = random.choice(urls)
                response = requests.get(url, timeout=5)
                self.stats['http_requests'] += 1
                print(f"📡 HTTP {response.status_code}: {url}")
                time.sleep(random.uniform(1, 3))
            except Exception as e:
                print(f"❌ HTTP error: {e}")
                time.sleep(2)

    def generate_tcp_traffic(self):
        """Generate TCP connections"""
        hosts = [
            ('8.8.8.8', 53),      # Google DNS
            ('1.1.1.1', 53),      # Cloudflare DNS
            ('github.com', 80),    # GitHub
            ('google.com', 80),    # Google
            ('stackoverflow.com', 80),  # StackOverflow
        ]
        
        while self.running:
            try:
                host, port = random.choice(hosts)
                
                # Resolve hostname to IP if needed
                try:
                    ip = socket.gethostbyname(host)
                except:
                    ip = host
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((ip, port))
                
                if result == 0:
                    # Send some data if it's HTTP
                    if port == 80:
                        sock.send(b"GET / HTTP/1.1\r\nHost: " + host.encode() + b"\r\n\r\n")
                        sock.recv(1024)  # Receive response
                    
                    self.stats['tcp_connections'] += 1
                    print(f"🔌 TCP connection to {host}:{port} ({ip})")
                
                sock.close()
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                print(f"❌ TCP error: {e}")
                time.sleep(3)

    def generate_udp_traffic(self):
        """Generate UDP traffic"""
        dns_servers = [
            ('8.8.8.8', 53),
            ('1.1.1.1', 53),
            ('208.67.222.222', 53),  # OpenDNS
        ]
        
        domains = [
            'google.com',
            'github.com',
            'stackoverflow.com',
            'python.org',
            'microsoft.com'
        ]
        
        while self.running:
            try:
                server, port = random.choice(dns_servers)
                domain = random.choice(domains)
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(3)
                
                # Simple DNS query (not a proper DNS packet, but generates UDP traffic)
                query = f"DNS query for {domain}".encode()
                sock.sendto(query, (server, port))
                
                try:
                    sock.recv(1024)
                except:
                    pass  # May not get a proper response
                
                sock.close()
                
                self.stats['udp_packets'] += 1
                self.stats['dns_queries'] += 1
                print(f"📡 UDP DNS query: {domain} via {server}")
                
                time.sleep(random.uniform(3, 6))
                
            except Exception as e:
                print(f"❌ UDP error: {e}")
                time.sleep(3)

    def generate_mixed_traffic(self):
        """Generate mixed traffic patterns"""
        patterns = [
            self.burst_traffic,
            self.slow_traffic,
            self.random_traffic
        ]
        
        while self.running:
            pattern = random.choice(patterns)
            print(f"🔄 Switching to {pattern.__name__}")
            pattern()

    def burst_traffic(self):
        """Generate burst of traffic"""
        print("💥 Generating traffic burst...")
        for _ in range(random.randint(5, 15)):
            if not self.running:
                break
            threading.Thread(target=self._single_request, daemon=True).start()
            time.sleep(0.1)
        time.sleep(random.uniform(5, 10))

    def slow_traffic(self):
        """Generate slow, steady traffic"""
        print("🐌 Generating slow traffic...")
        for _ in range(random.randint(3, 8)):
            if not self.running:
                break
            self._single_request()
            time.sleep(random.uniform(2, 5))

    def random_traffic(self):
        """Generate random traffic"""
        print("🎲 Generating random traffic...")
        for _ in range(random.randint(5, 20)):
            if not self.running:
                break
            self._single_request()
            time.sleep(random.uniform(0.5, 3))

    def _single_request(self):
        """Make a single HTTP request"""
        urls = [
            'http://httpbin.org/bytes/1024',
            'http://httpbin.org/delay/1',
            'https://api.github.com/users/octocat',
            'https://jsonplaceholder.typicode.com/posts'
        ]
        
        try:
            url = random.choice(urls)
            requests.get(url, timeout=5)
            self.stats['http_requests'] += 1
        except:
            pass

    def print_stats(self):
        """Print traffic statistics"""
        while self.running:
            time.sleep(10)
            print("\n" + "="*50)
            print("📊 TRAFFIC STATISTICS")
            print("="*50)
            for key, value in self.stats.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
            print("="*50)

    def run(self, duration=None):
        """Run the traffic generator"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                 Network Traffic Generator                    ║
║              Generating test traffic for                     ║
║                    IoT Sentinel                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        self.running = True
        threads = []
        
        # Start different traffic generators
        generators = [
            self.generate_http_traffic,
            self.generate_tcp_traffic,
            self.generate_udp_traffic,
        ]
        
        print("🚀 Starting traffic generators...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit traffic generators
            futures = [executor.submit(gen) for gen in generators]
            
            # Submit stats printer
            stats_future = executor.submit(self.print_stats)
            
            try:
                if duration:
                    print(f"🕒 Running for {duration} seconds...")
                    time.sleep(duration)
                else:
                    print("🕒 Running indefinitely (Press Ctrl+C to stop)...")
                    while True:
                        time.sleep(1)
                        
            except KeyboardInterrupt:
                print("\n🛑 Stopping traffic generator...")
            finally:
                self.running = False
                
                # Wait a bit for threads to finish
                time.sleep(2)
                
                print("\n📊 Final Statistics:")
                for key, value in self.stats.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                
                print("👋 Traffic generator stopped")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate network traffic for IoT Sentinel testing')
    parser.add_argument('--duration', '-d', type=int, help='Duration in seconds (default: run indefinitely)')
    parser.add_argument('--mode', '-m', choices=['http', 'tcp', 'udp', 'mixed'], 
                       default='mixed', help='Traffic generation mode')
    
    args = parser.parse_args()
    
    generator = TrafficGenerator()
    
    if args.mode != 'mixed':
        print(f"🎯 Running in {args.mode.upper()} mode only")
        generator.running = True
        
        mode_map = {
            'http': generator.generate_http_traffic,
            'tcp': generator.generate_tcp_traffic,
            'udp': generator.generate_udp_traffic
        }
        
        try:
            threading.Thread(target=generator.print_stats, daemon=True).start()
            mode_map[args.mode]()
        except KeyboardInterrupt:
            generator.running = False
            print("\n🛑 Stopped")
    else:
        generator.run(args.duration)

if __name__ == "__main__":
    main()