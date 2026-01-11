import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Network Intrusion Detection AI",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Network Intrusion Detection AI")
st.caption("Enter packet / flow details to detect malicious activity")

# Initialize session state for form values
if "src_port" not in st.session_state:
    st.session_state.src_port = 12302
if "dst_port" not in st.session_state:
    st.session_state.dst_port = 80
if "protocol" not in st.session_state:
    st.session_state.protocol = "TCP"
if "protocol_encoded" not in st.session_state:
    st.session_state.protocol_encoded = 6
if "l7_proto" not in st.session_state:
    st.session_state.l7_proto = "HTTP"
if "l7_proto_encoded" not in st.session_state:
    st.session_state.l7_proto_encoded = 1
if "tcp_flags" not in st.session_state:
    st.session_state.tcp_flags = "ACK"
if "tcp_flags_encoded" not in st.session_state:
    st.session_state.tcp_flags_encoded = 2
if "in_bytes" not in st.session_state:
    st.session_state.in_bytes = 1000
if "out_bytes" not in st.session_state:
    st.session_state.out_bytes = 2000
if "in_pkts" not in st.session_state:
    st.session_state.in_pkts = 10
if "out_pkts" not in st.session_state:
    st.session_state.out_pkts = 15
if "flow_duration" not in st.session_state:
    st.session_state.flow_duration = 5000
if "duration_in" not in st.session_state:
    st.session_state.duration_in = 100
if "min_ttl" not in st.session_state:
    st.session_state.min_ttl = 64
if "max_ttl" not in st.session_state:
    st.session_state.max_ttl = 64
if "longest_pkt" not in st.session_state:
    st.session_state.longest_pkt = 1500
if "shortest_pkt" not in st.session_state:
    st.session_state.shortest_pkt = 50
if "tcp_win_max_in" not in st.session_state:
    st.session_state.tcp_win_max_in = 0
if "tcp_win_max_out" not in st.session_state:
    st.session_state.tcp_win_max_out = 0
if "num_pkts_up_to_128" not in st.session_state:
    st.session_state.num_pkts_up_to_128 = 0
if "icmp_type" not in st.session_state:
    st.session_state.icmp_type = 0
if "icmp_ipv4_type" not in st.session_state:
    st.session_state.icmp_ipv4_type = 0
if "tcp_win_max_in" not in st.session_state:
    st.session_state.tcp_win_max_in = 0
if "tcp_win_max_out" not in st.session_state:
    st.session_state.tcp_win_max_out = 0
if "num_pkts_up_to_128" not in st.session_state:
    st.session_state.num_pkts_up_to_128 = 0
if "icmp_type" not in st.session_state:
    st.session_state.icmp_type = 0
if "icmp_ipv4_type" not in st.session_state:
    st.session_state.icmp_ipv4_type = 0

# Function to generate random packet details
def generate_random_packet():
    st.session_state.src_port = random.randint(1024, 65535)
    st.session_state.dst_port = random.randint(1, 65535)
    st.session_state.protocol = random.choice(["TCP", "UDP", "ICMP"])
    st.session_state.l7_proto = random.choice(["HTTP", "HTTPS", "DNS", "FTP", "OTHER"])
    st.session_state.tcp_flags = random.choice(["NONE", "SYN", "ACK", "FIN", "RST"])
    st.session_state.in_bytes = random.randint(100, 100000)
    st.session_state.out_bytes = random.randint(100, 100000)
    st.session_state.in_pkts = random.randint(1, 1000)
    st.session_state.out_pkts = random.randint(1, 1000)
    st.session_state.flow_duration = random.randint(100, 60000)
    st.session_state.duration_in = random.randint(10, 5000)
    st.session_state.min_ttl = random.randint(32, 255)
    st.session_state.max_ttl = random.randint(st.session_state.min_ttl, 255)
    st.session_state.longest_pkt = random.randint(1000, 1500)
    st.session_state.shortest_pkt = random.randint(20, 100)

@st.cache_resource
def load_benign_examples():
    """Load real benign examples from the dataset"""
    df = pd.read_csv('data/NF-UNSW-NB15-v2.csv')
    benign_rows = []
    
    # Get model to test confidence
    model = joblib.load('models/stage1_rf.pkl')
    feature_names = model.feature_names_in_
    
    # Find many benign examples (Label==0) that are confidently benign
    benign_df = df[df['Label'] == 0].copy()
    for idx, row in benign_df.iterrows():
        X = pd.DataFrame(0, index=[0], columns=feature_names)
        for col in feature_names:
            if col in row.index:
                X[col] = row[col]
        X = X.astype(np.float32)
        
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]  # Probability of being malicious
        
        if pred == 0 and prob <= 0.01:  # Confident benign (malicious prob < 1%)
            benign_rows.append(row)
            if len(benign_rows) >= 50:  # Load many examples
                break
    
    return benign_rows

def generate_benign_packet():
    """Populate form with real benign packet from dataset"""
    benign_examples = load_benign_examples()
    row = random.choice(benign_examples)
    
    # Map CSV columns to form fields
    st.session_state.src_port = int(row['L4_SRC_PORT'])
    st.session_state.dst_port = int(row['L4_DST_PORT'])
    st.session_state.protocol_encoded = int(row['PROTOCOL'])
    st.session_state.l7_proto_encoded = int(row['L7_PROTO'])
    st.session_state.tcp_flags_encoded = int(row['TCP_FLAGS'])
    
    st.session_state.in_bytes = int(row['IN_BYTES'])
    st.session_state.in_pkts = int(row['IN_PKTS'])
    st.session_state.out_bytes = int(row['OUT_BYTES'])
    st.session_state.out_pkts = int(row['OUT_PKTS'])
    
    st.session_state.flow_duration = int(row['FLOW_DURATION_MILLISECONDS'])
    st.session_state.duration_in = int(row['DURATION_IN'])
    
    st.session_state.min_ttl = int(row['MIN_TTL'])
    st.session_state.max_ttl = int(row['MAX_TTL'])
    st.session_state.longest_pkt = int(row['LONGEST_FLOW_PKT'])
    st.session_state.shortest_pkt = int(row['SHORTEST_FLOW_PKT'])
    
    st.session_state.tcp_win_max_in = int(row['TCP_WIN_MAX_IN'])
    st.session_state.tcp_win_max_out = int(row['TCP_WIN_MAX_OUT'])
    st.session_state.num_pkts_up_to_128 = int(row['NUM_PKTS_UP_TO_128_BYTES'])
    st.session_state.icmp_type = int(row['ICMP_TYPE'])
    st.session_state.icmp_ipv4_type = int(row['ICMP_IPV4_TYPE'])
    
    # Map protocol number to name
    protocol_map = {6: 'TCP', 17: 'UDP', 89: 'ICMP'}
    st.session_state.protocol = protocol_map.get(int(row['PROTOCOL']), 'TCP')
    
    # Map L7_PROTO number to name
    l7_map = {0: 'Other/Unknown', 1: 'HTTP', 2: 'DNS', 3: 'FTP', 4: 'HTTPS', 7: 'HTTP', 10: 'Other/Unknown', 131: 'Other/Unknown'}
    st.session_state.l7_proto = l7_map.get(int(row['L7_PROTO']), 'Other/Unknown')
    
    # Map TCP flags number to name
    tcp_flags_map = {0: 'NONE', 2: 'ACK', 18: 'SYN+ACK', 19: 'RST+ACK', 24: 'FIN', 25: 'FIN+ACK', 16: 'Other', 27: 'Other'}
    st.session_state.tcp_flags = tcp_flags_map.get(int(row['TCP_FLAGS']), 'Other')

# Function to generate MALICIOUS packet (actual examples from trained dataset)
@st.cache_resource
def load_malicious_examples():
    """Load pre-saved strictly malicious examples (100% confidence)"""
    df = pd.read_csv('data/strictly_malicious_packets.csv')
    return [row for _, row in df.iterrows()]

def generate_malicious_packet():
    """Populate form with real 100% malicious packet from dataset"""
    malicious_examples = load_malicious_examples()
    row = random.choice(malicious_examples)
    
    # Map CSV columns to form fields
    st.session_state.src_port = int(row['L4_SRC_PORT'])
    st.session_state.dst_port = int(row['L4_DST_PORT'])
    st.session_state.protocol_encoded = int(row['PROTOCOL'])
    st.session_state.l7_proto_encoded = int(row['L7_PROTO'])
    st.session_state.tcp_flags_encoded = int(row['TCP_FLAGS'])
    
    st.session_state.in_bytes = int(row['IN_BYTES'])
    st.session_state.in_pkts = int(row['IN_PKTS'])
    st.session_state.out_bytes = int(row['OUT_BYTES'])
    st.session_state.out_pkts = int(row['OUT_PKTS'])
    
    st.session_state.flow_duration = int(row['FLOW_DURATION_MILLISECONDS'])
    st.session_state.duration_in = int(row['DURATION_IN'])
    
    st.session_state.min_ttl = int(row['MIN_TTL'])
    st.session_state.max_ttl = int(row['MAX_TTL'])
    st.session_state.longest_pkt = int(row['LONGEST_FLOW_PKT'])
    st.session_state.shortest_pkt = int(row['SHORTEST_FLOW_PKT'])
    
    st.session_state.tcp_win_max_in = int(row['TCP_WIN_MAX_IN'])
    st.session_state.tcp_win_max_out = int(row['TCP_WIN_MAX_OUT'])
    st.session_state.num_pkts_up_to_128 = int(row['NUM_PKTS_UP_TO_128_BYTES'])
    st.session_state.icmp_type = int(row['ICMP_TYPE'])
    st.session_state.icmp_ipv4_type = int(row['ICMP_IPV4_TYPE'])
    
    # Map protocol number to name
    protocol_map = {6: 'TCP', 17: 'UDP', 89: 'ICMP'}
    st.session_state.protocol = protocol_map.get(int(row['PROTOCOL']), 'TCP')
    
    # Map L7_PROTO number to name (basic mapping)
    l7_map = {0: 'Other/Unknown', 1: 'HTTP', 2: 'DNS', 3: 'FTP', 4: 'HTTPS', 7: 'HTTP', 10: 'Other/Unknown', 131: 'Other/Unknown'}
    st.session_state.l7_proto = l7_map.get(int(row['L7_PROTO']), 'Other/Unknown')
    
    # Map TCP flags number to name
    tcp_flags_map = {0: 'NONE', 2: 'ACK', 18: 'SYN+ACK', 19: 'RST+ACK', 24: 'FIN', 25: 'FIN+ACK', 16: 'Other', 27: 'Other'}
    st.session_state.tcp_flags = tcp_flags_map.get(int(row['TCP_FLAGS']), 'Other')

# Buttons to generate different packet types
st.markdown("### 🎯 Quick Packet Generation")
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("🔀 Random Generate", use_container_width=True):
        generate_malicious_packet()
        st.session_state.form_submitted = False
        st.rerun()

with btn_col2:
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.src_port = 12302
        st.session_state.dst_port = 80
        st.session_state.protocol = "TCP"
        st.session_state.l7_proto = "HTTP"
        st.session_state.tcp_flags = "ACK"
        st.session_state.in_bytes = 1000
        st.session_state.out_bytes = 2000
        st.session_state.in_pkts = 10
        st.session_state.out_pkts = 15
        st.session_state.flow_duration = 5000
        st.session_state.duration_in = 100
        st.session_state.min_ttl = 64
        st.session_state.max_ttl = 64
        st.session_state.longest_pkt = 1500
        st.session_state.shortest_pkt = 50
        st.session_state.form_submitted = False
        st.rerun()

st.divider()

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "models/stage1_rf.pkl"   # adjust if name differs
model = joblib.load(MODEL_PATH)

# -----------------------------
# FEATURE LIST (MUST MATCH TRAINING)
# -----------------------------
FEATURES = [
    'L4_SRC_PORT','L4_DST_PORT','PROTOCOL','L7_PROTO',
    'IN_BYTES','IN_PKTS','OUT_BYTES','OUT_PKTS',
    'TCP_FLAGS','CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS',
    'FLOW_DURATION_MILLISECONDS','DURATION_IN',
    'MIN_TTL','MAX_TTL','LONGEST_FLOW_PKT','SHORTEST_FLOW_PKT',
    'MIN_IP_PKT_LEN','MAX_IP_PKT_LEN',
    'SRC_TO_DST_SECOND_BYTES','DST_TO_SRC_SECOND_BYTES',
    'SRC_TO_DST_AVG_THROUGHPUT','DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES',
    'TCP_WIN_MAX_IN','TCP_WIN_MAX_OUT',
    'ICMP_TYPE','ICMP_IPV4_TYPE',
    'DNS_QUERY_ID','DNS_QUERY_TYPE','DNS_TTL_ANSWER',
    'FTP_COMMAND_RET_CODE'
]

# -----------------------------
# SIMPLE CATEGORY ENCODING
# -----------------------------
PROTOCOL_MAP = {"TCP": 6, "UDP": 17, "ICMP": 89}
L7_MAP = {"HTTP": 1, "HTTPS": 4, "DNS": 2, "FTP": 3, "Other/Unknown": 0}
TCP_FLAG_MAP = {"NONE": 0, "ACK": 2, "SYN+ACK": 18, "RST+ACK": 19, "FIN": 24, "FIN+ACK": 25}

# -----------------------------
# USER INPUT FORM
# Get current values from session state
current_protocol = st.session_state.get("protocol", "TCP")
current_l7_proto = st.session_state.get("l7_proto", "HTTP")
current_tcp_flags = st.session_state.get("tcp_flags", "ACK")

# Handle protocol dropdown index
protocol_options = list(PROTOCOL_MAP.keys())
try:
    protocol_index = protocol_options.index(current_protocol)
except (ValueError, IndexError):
    protocol_index = 0

# Handle L7_PROTO dropdown index
l7_options = list(L7_MAP.keys())
try:
    l7_index = l7_options.index(current_l7_proto)
except (ValueError, IndexError):
    l7_index = 0

# Handle TCP_FLAGS dropdown index
tcp_flag_options = list(TCP_FLAG_MAP.keys())
try:
    tcp_flag_index = tcp_flag_options.index(current_tcp_flags)
except (ValueError, IndexError):
    tcp_flag_index = 0

with st.form("packet_form"):
    st.subheader("📋 Packet / Flow Input")

    src_port = st.number_input("Source Port", 0, 65535, value=st.session_state.src_port)
    dst_port = st.number_input("Destination Port", 0, 65535, value=st.session_state.dst_port)

    protocol = st.selectbox("Protocol", protocol_options, index=protocol_index)
    l7_proto = st.selectbox("L7 Protocol", l7_options, index=l7_index)
    tcp_flags = st.selectbox("TCP Flags", tcp_flag_options, index=tcp_flag_index)

    in_bytes = st.number_input("Incoming Bytes", 0, value=st.session_state.in_bytes)
    out_bytes = st.number_input("Outgoing Bytes", 0, value=st.session_state.out_bytes)
    
    in_pkts = st.number_input("Incoming Packets", 0, value=st.session_state.in_pkts)
    out_pkts = st.number_input("Outgoing Packets", 0, value=st.session_state.out_pkts)

    flow_duration = st.number_input("Flow Duration (ms)", 0, value=st.session_state.flow_duration)
    duration_in = st.number_input("Duration In", 0, value=st.session_state.duration_in)

    min_ttl = st.number_input("Min TTL", 0, value=st.session_state.min_ttl)
    max_ttl = st.number_input("Max TTL", 0, value=st.session_state.max_ttl)

    longest_pkt = st.number_input("Longest Packet (bytes)", 0, value=st.session_state.longest_pkt)
    shortest_pkt = st.number_input("Shortest Packet (bytes)", 0, value=st.session_state.shortest_pkt)
    
    # Additional TCP/ICMP fields
    tcp_win_max_in = st.number_input("TCP Window Max In", 0, value=st.session_state.tcp_win_max_in)
    tcp_win_max_out = st.number_input("TCP Window Max Out", 0, value=st.session_state.tcp_win_max_out)
    num_pkts_128 = st.number_input("Packets up to 128 bytes", 0, value=st.session_state.num_pkts_up_to_128)
    icmp_type = st.number_input("ICMP Type", 0, value=st.session_state.icmp_type)
    icmp_ipv4_type = st.number_input("ICMP IPv4 Type", 0, value=st.session_state.icmp_ipv4_type)

    submit = st.form_submit_button("🔍 Predict", use_container_width=True)

# -----------------------------
# PREDICTION
# -----------------------------
if submit:

    # Create empty feature row
    X = pd.DataFrame([{f: 0 for f in FEATURES}])

    # Fill user-provided values
    X['L4_SRC_PORT'] = src_port
    X['L4_DST_PORT'] = dst_port
    X['PROTOCOL'] = st.session_state.protocol_encoded
    X['L7_PROTO'] = st.session_state.l7_proto_encoded
    X['TCP_FLAGS'] = st.session_state.tcp_flags_encoded
    X['CLIENT_TCP_FLAGS'] = st.session_state.tcp_flags_encoded
    X['SERVER_TCP_FLAGS'] = st.session_state.tcp_flags_encoded

    X['IN_BYTES'] = in_bytes
    X['OUT_BYTES'] = out_bytes
    X['IN_PKTS'] = in_pkts
    X['OUT_PKTS'] = out_pkts
    X['FLOW_DURATION_MILLISECONDS'] = flow_duration
    X['DURATION_IN'] = duration_in

    X['MIN_TTL'] = min_ttl
    X['MAX_TTL'] = max_ttl
    X['LONGEST_FLOW_PKT'] = longest_pkt
    X['SHORTEST_FLOW_PKT'] = shortest_pkt
    X['MIN_IP_PKT_LEN'] = shortest_pkt
    X['MAX_IP_PKT_LEN'] = longest_pkt
    
    # Additional fields
    X['NUM_PKTS_UP_TO_128_BYTES'] = num_pkts_128
    X['TCP_WIN_MAX_IN'] = tcp_win_max_in
    X['TCP_WIN_MAX_OUT'] = tcp_win_max_out
    X['ICMP_TYPE'] = icmp_type
    X['ICMP_IPV4_TYPE'] = icmp_ipv4_type
    X['DNS_QUERY_ID'] = 0
    X['DNS_QUERY_TYPE'] = 0
    X['DNS_TTL_ANSWER'] = 0
    X['FTP_COMMAND_RET_CODE'] = 0
    
    # Compute derived features
    if flow_duration > 0:
        X['SRC_TO_DST_SECOND_BYTES'] = (in_bytes / flow_duration) * 1000
        X['DST_TO_SRC_SECOND_BYTES'] = (out_bytes / flow_duration) * 1000
        X['SRC_TO_DST_AVG_THROUGHPUT'] = (in_bytes * 8 / flow_duration) * 1000
        X['DST_TO_SRC_AVG_THROUGHPUT'] = (out_bytes * 8 / flow_duration) * 1000
    else:
        X['SRC_TO_DST_SECOND_BYTES'] = in_bytes
        X['DST_TO_SRC_SECOND_BYTES'] = out_bytes
        X['SRC_TO_DST_AVG_THROUGHPUT'] = in_bytes * 8000
        X['DST_TO_SRC_AVG_THROUGHPUT'] = out_bytes * 8000

    # Convert everything to float
    X = X.astype(np.float32)
    
    # Predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    
    st.divider()
    if pred == 1:
        st.error(f"🚨 **MALICIOUS TRAFFIC DETECTED**\n\nConfidence: {prob:.2%}")
    else:
        st.success(f"✅ **BENIGN TRAFFIC**\n\nConfidence: {(1 - prob):.2%}")

