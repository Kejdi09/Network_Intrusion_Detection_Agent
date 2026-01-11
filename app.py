import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
import seaborn as sns
from src.intelligent_agent import ExpertSystem, PlanningAgent, AdaptivelearningSystem

# Set seaborn style for better looking plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 5)

# Initialize Intelligent Agent Systems
@st.cache_resource
def load_expert_system():
    """Load the expert system for intelligent decision making"""
    return ExpertSystem()

@st.cache_resource
def load_planning_agent():
    """Load the planning agent for action sequencing"""
    return PlanningAgent()

@st.cache_resource
def load_learning_system():
    """Load the adaptive learning system"""
    return AdaptivelearningSystem()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Network Intrusion Detection AI",
    page_icon="🛡️",
    layout="wide"
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
    """Load pre-saved sample generator examples for sample generation"""
    try:
        df = pd.read_csv('data/sample_generator_data.csv')
    except FileNotFoundError:
        # Fallback: create synthetic packet if data file not available
        df = pd.DataFrame({'L4_SRC_PORT': [443], 'L4_DST_PORT': [12345], 'PROTOCOL': [6], 
                          'L7_PROTO': [91], 'TCP_FLAGS': [2], 'IN_BYTES': [5000000], 
                          'IN_PKTS': [100000], 'OUT_BYTES': [100], 'OUT_PKTS': [2], 
                          'DURATION_IN': [60], 'DURATION_OUT': [60], 'MIN_TTL': [64], 
                          'MAX_TTL': [64]})
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

@st.cache_resource
def load_anomaly_model():
    """Load the anomaly detection model (Isolation Forest)"""
    return joblib.load('models/anomaly_iforest.pkl')

@st.cache_resource
def load_stage1_models():
    """Load Stage 1 models (benign/malicious classification)"""
    # Try to load ensemble first, then fall back to RF
    try:
        return joblib.load('models/stage1_ensemble.pkl')
    except:
        try:
            return joblib.load('models/stage1_rf.pkl')
        except:
            return None

@st.cache_resource
def load_stage2_models():
    """Load Stage 2 models (attack type classification)"""
    try:
        ensemble, encoder = joblib.load('models/stage2_ensemble.pkl')
        return ensemble, encoder
    except:
        try:
            xgb, encoder = joblib.load('models/stage2_xgb.pkl')
            return xgb, encoder
        except:
            try:
                rf, encoder = joblib.load('models/stage2_rf.pkl')
                return rf, encoder
            except:
                return None, None

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
# LOAD MODELS
# -----------------------------
anomaly_model = load_anomaly_model()
stage1_model = load_stage1_models()
stage2_model, stage2_encoder = load_stage2_models()

# Model info
st.info("🧠 **System Architecture**: 3-Stage Detection Pipeline\n\n1️⃣ **Anomaly Detection** - Isolation Forest flags suspicious patterns\n2️⃣ **Stage 1 Classification** - Determines if traffic is benign/malicious\n3️⃣ **Stage 2 Classification** - Identifies specific attack type if malicious")

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
    protocol_encoded = PROTOCOL_MAP[protocol]  # Encode selected protocol
    
    l7_proto = st.selectbox("L7 Protocol", l7_options, index=l7_index)
    l7_proto_encoded = L7_MAP[l7_proto]  # Encode selected L7 protocol
    
    tcp_flags = st.selectbox("TCP Flags", tcp_flag_options, index=tcp_flag_index)
    tcp_flags_encoded = TCP_FLAG_MAP[tcp_flags]  # Encode selected TCP flags

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

# ============================================
# PREDICTION AND TWO-STAGE DETECTION PIPELINE
# ============================================
if submit:
    # Create empty feature row
    X = pd.DataFrame([{f: 0 for f in FEATURES}])

    # Fill user-provided values
    X['L4_SRC_PORT'] = src_port
    X['L4_DST_PORT'] = dst_port
    X['PROTOCOL'] = protocol_encoded  # Use encoded value from form
    X['L7_PROTO'] = l7_proto_encoded  # Use encoded value from form
    X['TCP_FLAGS'] = tcp_flags_encoded  # Use encoded value from form
    X['CLIENT_TCP_FLAGS'] = tcp_flags_encoded
    X['SERVER_TCP_FLAGS'] = tcp_flags_encoded

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
    
    st.divider()
    
    # ============================================
    # STAGE 0: ANOMALY DETECTION
    # ============================================
    anomaly_score = -anomaly_model.decision_function(X)[0]
    anomaly_threshold = 0.5
    is_anomaly = anomaly_score >= anomaly_threshold
    
    # ============================================
    # STAGE 1: BENIGN VS MALICIOUS CLASSIFICATION
    # ============================================
    stage1_pred = stage1_model.predict(X)[0]
    stage1_proba = stage1_model.predict_proba(X)[0]
    benign_prob = stage1_proba[0]
    malicious_prob = stage1_proba[1]
    
    # Boost confidence based on anomaly score if anomaly detected
    if is_anomaly and stage1_pred == 1:
        malicious_prob = min(1.0, malicious_prob + 0.15)
        benign_prob = max(0.0, benign_prob - 0.15)
    
    # Get attack type if malicious
    attack_type = None
    attack_confidence = 0.0
    if stage1_pred == 1 and stage2_model is not None and stage2_encoder is not None:
        stage2_pred = stage2_model.predict(X)[0]
        stage2_proba = stage2_model.predict_proba(X)
        attack_type = stage2_encoder.inverse_transform([stage2_pred])[0]
        attack_confidence = max(stage2_proba[0])
    
    # ============================================
    # INTELLIGENT AGENT DECISION MAKING
    # ============================================
    expert_system = load_expert_system()
    planning_agent = load_planning_agent()
    learning_system = load_learning_system()
    
    # Get expert system decision
    agent_decision = expert_system.evaluate(
        malicious_prob=malicious_prob,
        benign_prob=benign_prob,
        anomaly_score=anomaly_score,
        attack_type=attack_type,
        attack_confidence=attack_confidence
    )
    
    # Display detection summary
    st.subheader("🔍 Detection Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Anomaly Score", f"{anomaly_score:.4f}")
    with col2:
        st.metric("Benign Probability", f"{benign_prob:.2%}")
    with col3:
        st.metric("Malicious Probability", f"{malicious_prob:.2%}")
    
    # ============================================
    # INTELLIGENT AGENT RECOMMENDATION
    # ============================================
    st.divider()
    st.subheader("🤖 Intelligent Agent Analysis")
    
    # Display decision with reasoning
    col_decision, col_threat = st.columns([2, 1])
    
    with col_decision:
        st.write("**Agent Decision:**")
        st.write(f"**Action**: {agent_decision.action.value.upper()}")
        
    with col_threat:
        threat_colors = {
            "SAFE": "🟢",
            "LOW": "🟡",
            "MEDIUM": "🟠",
            "HIGH": "🔴",
            "CRITICAL": "🔴"
        }
        color = threat_colors.get(agent_decision.threat_level.name, "⚪")
        st.write("**Threat Level:**")
        st.write(f"{color} {agent_decision.threat_level.name}")
    
    st.info(f"**Agent Reasoning:**\n\n{agent_decision.explanation}")
    
    # ============================================
    # DECISION LOGIC
    # ============================================
    
    if stage1_pred == 0:  # Benign
        st.success(f"✅ **BENIGN TRAFFIC DETECTED**\n\n**Confidence**: {benign_prob:.2%}")
        st.success(f"**Agent Recommendation**: {agent_decision.action.value.upper()}")
        st.info(f"**Analysis Summary:**\n\n- Anomaly Score: {anomaly_score:.6f}\n- Benign Classification: {benign_prob:.2%}\n- Status: ✓ Safe to allow")
        
    else:  # Malicious
        st.error(f"🚨 **MALICIOUS TRAFFIC DETECTED**\n\n**Confidence**: {malicious_prob:.2%}")
        st.error(f"**Agent Recommendation**: {agent_decision.action.value.upper()}")
        
        # ============================================
        # STAGE 2: ATTACK TYPE CLASSIFICATION
        # ============================================
        if attack_type is not None:
            
            # Enhanced attack type display
            st.subheader("🎯 Attack Type Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Detected Attack", attack_type)
            with col2:
                st.metric("Attack Confidence", f"{attack_confidence:.2%}")
            with col3:
                st.metric("Threat Level", "🔴 CRITICAL" if attack_confidence > 0.85 else "🟠 HIGH" if attack_confidence > 0.70 else "🟡 MEDIUM")
            
            # Attack description
            attack_descriptions = {
                'Botnet': 'Bot-infected systems sending malicious traffic. Origin: Compromised devices in a network.',
                'DoS': 'Denial of Service - Overwhelming target with traffic to cause unavailability.',
                'DoS-UDP': 'UDP-based Denial of Service attack. Floods target with UDP packets.',
                'DoS-TCP': 'TCP-based Denial of Service attack. Floods target with TCP packets.',
                'Backdoor': 'Unauthorized remote access mechanism installed on system.',
                'Worm': 'Self-propagating malicious software spreading through network.',
                'Exploit': 'Malicious code exploiting software vulnerabilities.',
                'Reconnaissance': 'Scanning and probing for system vulnerabilities.',
                'Shellcode': 'Code designed to spawn command shell for remote execution.',
                'Rootkit': 'Malicious software providing root/admin level access.',
                'Trojan': 'Malware disguised as legitimate software.',
                'Virus': 'Self-replicating malicious code.',
                'Ransomware': 'Malware encrypting files and demanding payment.',
                'Spyware': 'Software secretly collecting user information.',
                'DDoS': 'Distributed Denial of Service - Multiple sources overwhelming target.',
            }
            
            attack_desc = attack_descriptions.get(attack_type, f"Unknown attack type: {attack_type}")
            st.info(f"**What is {attack_type}?**\n\n{attack_desc}")
            
            st.warning(f"**Attack Type**: {attack_type}\n\n**Attack Confidence**: {attack_confidence:.2%}")
            
            # Attack type distribution chart
            st.subheader("📊 Attack Type Probability Distribution")
            attack_probs = {}
            for idx, attack_name in enumerate(stage2_encoder.classes_):
                attack_probs[attack_name] = stage2_proba[0][idx]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            attacks = list(attack_probs.keys())
            probs = list(attack_probs.values())
            colors = ['#e74c3c' if p == max(probs) else '#95a5a6' for p in probs]
            
            bars = ax.barh(attacks, probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
            ax.set_ylabel('Attack Type', fontsize=12, fontweight='bold')
            ax.set_title(f'Attack Type Classification: {attack_type} (Confidence: {attack_confidence:.2%})', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            
            for i, (attack, prob) in enumerate(zip(attacks, probs)):
                ax.text(prob + 0.02, i, f'{prob:.2%}', va='center', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed attack analysis
            st.subheader("🔍 Detailed Attack Analysis")
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.write("**Top 3 Detected Attacks:**")
                sorted_attacks = sorted(attack_probs.items(), key=lambda x: x[1], reverse=True)
                for idx, (atk_name, atk_prob) in enumerate(sorted_attacks[:3], 1):
                    st.write(f"{idx}. **{atk_name}**: {atk_prob:.2%}")
            
            with analysis_col2:
                st.write("**Attack Characteristics:**")
                st.write(f"- **Type**: {attack_type}")
                st.write(f"- **Confidence**: {attack_confidence:.2%}")
                st.write(f"- **Risk Level**: {'🔴 CRITICAL' if attack_confidence > 0.85 else '🟠 HIGH' if attack_confidence > 0.70 else '🟡 MEDIUM'}")
                st.write(f"- **Recommended Action**: Immediate isolation & investigation")
        
        st.error(f"**Analysis Summary:**\n\n- Anomaly Score: {anomaly_score:.6f}\n- Malicious Classification: {malicious_prob:.2%}\n- **⛔ Recommended Action: BLOCK IMMEDIATELY**")
    
    # ============================================
    # INTELLIGENT AGENT: REMEDIATION PLANNING
    # ============================================
    st.divider()
    st.subheader("📋 Remediation & Action Plan")
    
    # Display remediation plan from agent
    if agent_decision.remediation_plan:
        st.info("**Recommended Actions (in order):**")
        for idx, action in enumerate(agent_decision.remediation_plan, 1):
            st.write(f"{idx}. {action}")
    
    # Generate and display sequential response plan
    response_plan = planning_agent.generate_response_plan(
        agent_decision.threat_level,
        attack_type if attack_type else "Unknown",
        attack_confidence if attack_type else 0.0
    )
    
    st.write("\n**Sequential Response Plan:**")
    tab1, tab2 = st.tabs(["📈 Timeline View", "🎯 Action Details"])
    
    with tab1:
        plan_df = pd.DataFrame([
            {"Phase": idx, "Action": action[0], "Details": action[1]}
            for idx, action in enumerate(response_plan, 1)
        ])
        st.dataframe(plan_df, use_container_width=True)
    
    with tab2:
        for phase, (action, reasoning) in enumerate(response_plan, 1):
            with st.expander(f"Phase {phase}: {action}"):
                st.write(reasoning)
    
    # ============================================
    # ADAPTIVE LEARNING FEEDBACK
    # ============================================
    st.divider()
    st.subheader("🧠 Adaptive Learning System")
    
    col_feedback1, col_feedback2 = st.columns(2)
    
    with col_feedback1:
        st.write("**Was this prediction correct?**")
        feedback_correct = st.radio(
            "Select prediction accuracy:",
            ["Correct", "Incorrect"],
            key=f"feedback_{random.random()}"
        )
    
    with col_feedback2:
        st.write("**Confidence Assessment:**")
        agent_decision_str = "Malicious" if stage1_pred == 1 else "Benign"
        feedback_note = st.text_input(
            "Add notes (optional):",
            key=f"feedback_note_{random.random()}"
        )
    
    if st.button("📊 Submit Feedback", key=f"submit_feedback_{random.random()}"):
        learning_system.record_feedback(
            prediction=agent_decision_str,
            actual="Malicious" if feedback_correct == "Correct" and stage1_pred == 1 else "Benign",
            confidence=max(malicious_prob, benign_prob)
        )
        st.success("✓ Feedback recorded for model improvement!")
    
    # Display learning metrics
    st.write("**Learning System Performance:**")
    perf_report = learning_system.get_performance_report()
    
    if perf_report.get("total_samples", 0) > 0:
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Accuracy", f"{perf_report['accuracy']:.1%}")
        with metric_col2:
            st.metric("Samples Learned", perf_report['total_samples'])
        with metric_col3:
            st.metric("Avg Confidence", f"{perf_report['confidence_avg']:.1%}")
        
        if perf_report.get("suggested_adjustments"):
            st.warning("**Suggested Threshold Adjustments:**")
            for suggestion, detail in perf_report["suggested_adjustments"].items():
                st.write(f"- {detail}")
    else:
        st.info("ℹ️ System learning from feedback - more samples needed for threshold suggestions")
    
    # ============================================
    # VISUALIZATION: DETECTION PIPELINE
    # ============================================
    st.subheader("📊 Detection Pipeline Visualization")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Classification pie chart
    labels = ['Benign', 'Malicious']
    sizes = [benign_prob * 100, malicious_prob * 100]
    colors_pie = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)
    
    axes[0].pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    axes[0].set_title('Stage 1: Benign vs Malicious Classification', fontsize=13, fontweight='bold', pad=20)
    
    # Right: Confidence metrics
    decision_metrics = ['Benign\nProb', 'Malicious\nProb', 'Anomaly\nScore']
    decision_values = [benign_prob, malicious_prob, min(anomaly_score / 10, 1.0)]
    colors_bars = ['#2ecc71', '#e74c3c', '#f39c12']
    
    bars = axes[1].bar(decision_metrics, decision_values, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Score / Probability', fontsize=12, fontweight='bold')
    axes[1].set_title('Detection Confidence Metrics', fontsize=13, fontweight='bold', pad=20)
    axes[1].set_ylim(0, 1)
    axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Decision Threshold')
    
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ============================================
    # INPUT SUMMARY
    # ============================================
    st.subheader("📝 Input Packet Summary")
    summary_data = {
        'Ports': f"{src_port} → {dst_port}",
        'Protocol': protocol,
        'Traffic': f"In: {in_bytes}B ({in_pkts}pkt) | Out: {out_bytes}B ({out_pkts}pkt)",
        'Duration': f"{flow_duration}ms",
        'TTL Range': f"{min_ttl} - {max_ttl}",
        'Packet Size': f"{shortest_pkt}B - {longest_pkt}B"
    }
    
    for key, value in summary_data.items():
        st.write(f"**{key}**: {value}")

