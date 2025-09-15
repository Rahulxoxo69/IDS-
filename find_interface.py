import scapy.all as scapy

print("Available network interfaces:")
interfaces = scapy.get_if_list()
for i, interface in enumerate(interfaces):
    print(f"{i}: {interface}")

print("\nDefault interface:", scapy.conf.iface)