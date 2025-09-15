from flask import Blueprint, Response
from prometheus_client import Counter, Histogram, generate_latest

metrics_bp=Blueprint('metrics',__name__)

PKTS=Counter('ids_packets_total','Packets analysed')
THRTS=Counter('ids_threats_total','Threats detected')
LAT=Histogram('ids_predict_seconds','Prediction latency')

@metrics_bp.route('/metrics')
def metrics():
    return Response(generate_latest(),mimetype='text/plain')