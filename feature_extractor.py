# Replace the entire contents of feature_extractor.py with this:

import pandas as pd
import numpy as np
from scapy.all import IP, TCP, UDP, ICMP, Raw
from scipy.stats import entropy as scipy_entropy

class FeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'protocol', 'packet_length', 'ttl', 'tcp_flags',
            'window_size', 'payload_length', 'entropy',
            'payload_ratio', 'dst_port'
        ]

    def _entropy(self, data):
        if not data: return 0
        counts = np.bincount(bytearray(data))
        if len(counts) == 0: return 0
        probs = counts / len(data)
        return scipy_entropy(probs)

    def extract_features(self, packets):
        feats = []
        for pkt in packets:
            if not pkt.haslayer(IP): 
                continue
            
            # --- Protocol ---
            proto = 0 if pkt.haslayer(TCP) else 1 if pkt.haslayer(UDP) else 2 if pkt.haslayer(ICMP) else 3
            
            # --- Packet lengths and TTL ---
            pkt_len = len(pkt)
            ttl = pkt[IP].ttl
            payload_len = len(pkt[Raw]) if pkt.haslayer(Raw) else 0
            
            # --- Ports, Flags, and Window (default to 0) ---
            dst_port = pkt.dport if hasattr(pkt, 'dport') else 0
            window = pkt[TCP].window if pkt.haslayer(TCP) else 0
            flags = int(pkt[TCP].flags) if pkt.haslayer(TCP) else 0

            # --- Calculated Features ---
            ent = self._entropy(pkt[Raw].load) if pkt.haslayer(Raw) else 0
            payload_ratio = payload_len / pkt_len if pkt_len > 0 else 0

            # --- Append the completed feature set ---
            feats.append([
                proto, pkt_len, ttl, flags,
                window, payload_len, ent,
                payload_ratio, dst_port
            ])

        # --- Return the final DataFrame ---
        if not feats:
            return pd.DataFrame(columns=self.feature_names)
        return pd.DataFrame(feats, columns=self.feature_names)