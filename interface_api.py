from flask import Blueprint, jsonify
from scapy.all import get_if_list, sniff

iface_bp = Blueprint('iface', __name__, url_prefix='/api')

def test_capture(iface, count=3, timeout=2):
    try:
        return len(sniff(iface=iface, count=count, timeout=timeout, quiet=True)) > 0
    except Exception:
        return False

@iface_bp.route('/interfaces')
def interfaces():
    return jsonify([{"idx": i, "name": n, "working": test_capture(n)}
                    for i, n in enumerate(get_if_list())])