import socket
import json
import time
import random
from datetime import datetime
import threading

class IoTSimulator:
    def __init__(self, server_host='localhost', server_port=8888):
        self.server_host = server_host
        self.server_port = server_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.devices = []
        self.running = False
        
    def create_device(self, device_type, device_id=None):
        """Create a new IoT device simulator"""
        if device_id is None:
            device_id = f"{device_type}_{random.randint(1000, 9999)}"
        
        device = {
            'id': device_id,
            'type': device_type,
            'interval': random.uniform(1.0, 10.0),
            'last_sent': 0
        }
        
        self.devices.append(device)
        return device
    
    def generate_temperature_sensor_data(self, device_id):
        """Generate temperature sensor data"""
        base_temp = 20.0 + random.uniform(-2.0, 2.0)
        fluctuation = random.uniform(-1.0, 1.0)
        temperature = base_temp + fluctuation
        
        return {
            'device_id': device_id,
            'device_type': 'temperature_sensor',
            'timestamp': datetime.now().isoformat(),
            'temperature': round(temperature, 2),
            'unit': 'celsius',
            'battery': random.randint(20, 100),
            'location': f"Room_{random.randint(1, 10)}"
        }
    
    def generate_humidity_sensor_data(self, device_id):
        """Generate humidity sensor data"""
        base_humidity = 45.0 + random.uniform(-5.0, 5.0)
        fluctuation = random.uniform(-3.0, 3.0)
        humidity = max(10.0, min(90.0, base_humidity + fluctuation))
        
        return {
            'device_id': device_id,
            'device_type': 'humidity_sensor',
            'timestamp': datetime.now().isoformat(),
            'humidity': round(humidity, 2),
            'unit': 'percent',
            'battery': random.randint(20, 100),
            'location': f"Room_{random.randint(1, 10)}"
        }
    
    def generate_smart_light_data(self, device_id):
        """Generate smart light data"""
        states = ['on', 'off']
        colors = ['white', 'warm_white', 'red', 'green', 'blue', 'yellow']
        
        return {
            'device_id': device_id,
            'device_type': 'smart_light',
            'timestamp': datetime.now().isoformat(),
            'state': random.choice(states),
            'brightness': random.randint(0, 100),
            'color': random.choice(colors),
            'power_consumption': round(random.uniform(5.0, 15.0), 2)
        }
    
    def generate_security_camera_data(self, device_id):
        """Generate security camera data"""
        events = ['motion_detected', 'no_motion', 'sound_detected', 'face_detected']
        
        return {
            'device_id': device_id,
            'device_type': 'security_camera',
            'timestamp': datetime.now().isoformat(),
            'event': random.choice(events),
            'resolution': '1080p',
            'fps': random.randint(15, 30),
            'storage_remaining': round(random.uniform(10.0, 100.0), 2)
        }
    
    def generate_smart_lock_data(self, device_id):
        """Generate smart lock data"""
        states = ['locked', 'unlocked']
        events = ['locked', 'unlocked', 'auto_locked', 'access_denied', 'tamper_alert']
        
        return {
            'device_id': device_id,
            'device_type': 'smart_lock',
            'timestamp': datetime.now().isoformat(),
            'state': random.choice(states),
            'last_event': random.choice(events),
            'battery': random.randint(20, 100),
            'users': random.randint(1, 10)
        }
    
    def send_data(self, device):
        """Send data for a specific device"""
        try:
            if device['type'] == 'temperature_sensor':
                data = self.generate_temperature_sensor_data(device['id'])
            elif device['type'] == 'humidity_sensor':
                data = self.generate_humidity_sensor_data(device['id'])
            elif device['type'] == 'smart_light':
                data = self.generate_smart_light_data(device['id'])
            elif device['type'] == 'security_camera':
                data = self.generate_security_camera_data(device['id'])
            elif device['type'] == 'smart_lock':
                data = self.generate_smart_lock_data(device['id'])
            else:
                return
            
            json_data = json.dumps(data)
            self.socket.sendto(json_data.encode('utf-8'), (self.server_host, self.server_port))
            
            print(f"[IoT Simulator] Sent data from {device['id']}: {device['type']}")
            
        except Exception as e:
            print(f"[IoT Simulator] Error sending data for {device['id']}: {e}")
    
    def run(self, num_devices=10):
        """Run the IoT simulator with the specified number of devices"""
        self.running = True
        
        # Create IoT devices
        device_types = [
            'temperature_sensor', 
            'humidity_sensor', 
            'smart_light', 
            'security_camera', 
            'smart_lock'
        ]
        
        for i in range(num_devices):
            device_type = random.choice(device_types)
            self.create_device(device_type)
        
        print(f"[IoT Simulator] Started with {num_devices} devices")
        print(f"[IoT Simulator] Sending data to {self.server_host}:{self.server_port}")
        print("[IoT Simulator] Press Ctrl+C to stop...")
        
        # Main loop
        try:
            while self.running:
                current_time = time.time()
                
                for device in self.devices:
                    if current_time - device['last_sent'] >= device['interval']:
                        self.send_data(device)
                        device['last_sent'] = current_time
                        # Add some jitter to avoid all devices sending at once
                        time.sleep(random.uniform(0.01, 0.1))
                
                time.sleep(0.1)  # Small sleep to prevent CPU overload
                
        except KeyboardInterrupt:
            print("\n[IoT Simulator] Shutting down...")
            self.stop()
    
    def stop(self):
        """Stop the IoT simulator"""
        self.running = False
        if self.socket:
            self.socket.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='IoT Device Simulator')
    parser.add_argument('--host', default='localhost', help='Server hostname')
    parser.add_argument('--port', type=int, default=8888, help='Server port')
    parser.add_argument('--devices', type=int, default=10, help='Number of devices to simulate')
    
    args = parser.parse_args()
    
    simulator = IoTSimulator(server_host=args.host, server_port=args.port)
    simulator.run(num_devices=args.devices)

if __name__ == "__main__":
    main()