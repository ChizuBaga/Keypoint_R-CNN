import streamlit as st
import cv2
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
)

# GCN Specific imports
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# CUDA / CPU
device = "cpu"

# wEB config
st.set_page_config(
    page_title="Sitting Posture Classification",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

############################ 
# Generated from Claude
############################
# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: white; color: black; }
section[data-testid="stSidebar"] {
  border-right: 1px solid #181d28;
}

.status-good {
  border: 1.5px solid #22c55e;
  border-radius: 14px; padding: 20px 28px;
  text-align: center; font-size: 1.6rem; font-weight: 800;
  color: #22c55e; letter-spacing: 3px; text-transform: uppercase;
  box-shadow: 0 0 40px rgba(34,197,94,.15), inset 0 0 40px rgba(34,197,94,.04);
}
.status-bad {
  border: 1.5px solid #ef4444;
  border-radius: 14px; padding: 20px 28px;
  text-align: center; font-size: 1.6rem; font-weight: 800;
  color: #ef4444; letter-spacing: 3px; text-transform: uppercase;
  box-shadow: 0 0 40px rgba(239,68,68,.15), inset 0 0 40px rgba(239,68,68,.04);
}
.status-idle {
  border: 1px solid #1e2535;
  border-radius: 14px; padding: 20px 28px;
  text-align: center; font-size: 1rem; color: #4b5568; letter-spacing: 1px;
}
.card {
  border: 1px solid #181d28;
  border-radius: 10px; padding: 14px 18px; margin: 5px 0;
}
.card-label { font-size:.68rem;  letter-spacing:2px; text-transform:uppercase; margin-bottom:4px; }
.card-value { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:600; }
.bar-bg { border-radius:4px; height:6px; margin-top:8px; overflow:hidden; }
.bar-fill { height:100%; border-radius:4px; }
.kp-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:3px; margin-top:6px; }
.kp-chip {
  border:1px solid #181d28; border-radius:5px;
  padding:3px 7px; font-family:'JetBrains Mono',monospace;
  font-size:.6rem; color:#6b7a99; white-space:nowrap;
}
.kp-chip.active { border-color:#2563eb; color:#93c5fd; }
.stat-row { display:flex; gap:8px; margin-top:4px; }
.stat-pill { flex:1; border-radius:8px; padding:10px 8px; text-align:center; font-size:.7rem; letter-spacing:1px; text-transform:uppercase; }
.stat-good { background:#071a0f; border:1px solid #166534; color:#4ade80; }
.stat-bad  { background:#1a0707; border:1px solid #991b1b; color:#f87171; }
.block-container { padding-top:1.2rem; padding-bottom:.5rem; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)
############################ 
# End
############################

# COCO keypoint
parts = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

skeleton_edges = [
    [0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5,6], [5, 7], 
    [5, 11], [6, 12], [6, 8], [7, 9], [8, 10], [11, 12], [13, 11], 
    [14, 12], [15, 13], [16, 14]
]

# Keypoint R-CNN
def load_detector():
    model  = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.to(device).eval()
    return model

# Load sitting posture model
def load_posture_model(path: str, index_model: int):

    match index_model:
        # MLP
        case 0:
            class MLP(nn.Module):
                def __init__(self):
                    """
                    Create a Fully Connected Neuron Network

                    """
                    super().__init__()
                    self.layer1 = nn.Sequential(
                        nn.Linear(in_features=51, out_features=128),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.1),
                        nn.Dropout(0.5),
                    )
                    self.layer2 = nn.Sequential(
                        nn.Linear(in_features=128, out_features=64),
                        nn.BatchNorm1d(64),
                        nn.LeakyReLU(0.1),
                        nn.Dropout(0.2)
                    )
                    self.layer3 = nn.Sequential(
                        nn.Linear(in_features=64, out_features=32),
                        nn.BatchNorm1d(32),
                        nn.LeakyReLU(0.1),
                        nn.Dropout(0.2)
                    )
                    self.outputlayer = nn.Linear(in_features=32, out_features=1)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    """
                    To perform inference (do prediction).
                    During training mode, this will construct the computational graph
                    """
                    # perform forward propagation and build computational graph
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.outputlayer(x)
                    x = self.sigmoid(x)

                    return x
    
            net = MLP()
            net.load_state_dict(torch.load(path, map_location=device))
            net.to(device).eval()
            return net
        case 1:            
            class GCN_model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = GCNConv(3, 8)
                    self.bn1 = nn.BatchNorm1d(8)
                    self.conv2 = GCNConv(8, 16)
                    self.bn2 = nn.BatchNorm1d(16)

                    self.fc1 = nn.Linear(17 * 16, 32)
                    
                    self.dropout = nn.Dropout(0.5)
                    
                    self.fc2 = nn.Linear(32, 1)

                    self.relu = nn.ReLU()

                def forward(self, data):
                    x, edge_index = data.x, data.edge_index
                    x = self.relu(self.bn1(self.conv1(x, edge_index)))
                    x = self.relu(self.bn2(self.conv2(x, edge_index)))

                    ####################################
                    # Generated from Gemini
                    ####################################
                    
                    x = x.view(-1, 17 * 16)

                    ####################################
                    # End
                    ####################################
                    x = self.relu(self.fc1(x))
                    x = self.dropout(x)
                    out = self.fc2(x)
                    return out
                
            net = GCN_model()
            net.load_state_dict(torch.load(path, map_location=device))
            net.to(device).eval()
            return net
        case 2:
            class CNN1d(nn.Module):
                def __init__(self):
                    super().__init__()

                    self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
                    self.bn1 = nn.BatchNorm1d(32)
                    self.pool1 = nn.MaxPool1d(kernel_size=2)

                    self.spatial_dropout1 = nn.Dropout1d(p=0.2)

                    self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
                    self.bn2 = nn.BatchNorm1d(64)
                    self.pool2 = nn.MaxPool1d(kernel_size=2)

                    self.spatial_dropout2 = nn.Dropout1d(p=0.2)

                    self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

                    self.fc1 = nn.Linear(64, 32)
                    self.dropout = nn.Dropout(p=0.6)
                    self.fc2 = nn.Linear(32, 1)

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = torch.relu(x)
                    x = self.pool1(x)

                    x = self.spatial_dropout1(x)

                    x = self.conv2(x)
                    x = self.bn2(x)
                    x = torch.relu(x)
                    x = self.pool2(x)

                    x = self.spatial_dropout2(x)

                    x = self.global_avg_pool(x)
                    x = torch.flatten(x, 1)

                    x = self.fc1(x)
                    x = torch.relu(x)
                    x = self.dropout(x)

                    x = self.fc2(x)

                    return x
                
            net = CNN1d()
            net.load_state_dict(torch.load(path, map_location=device))
            net.to(device).eval()
            return net
        case 3:
            return None

# Extraction keypoint 
def extract_keypoint(img, model, device):
    """
    Returns (17,3) ndarray ==> [x, y, visibility].
    """
    img_for_drawing = F.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_for_drawing)[0]

    kp = output["keypoints"]
    scores = output["scores"].cpu().numpy()
    if len(scores) == 0:
        return None

    best = int(np.argmax(scores)) # Highest scores
    if scores[best] < 0.9:
        return None # No human detected

    return kp[best].detach().cpu().numpy()

def normalize_coco_posture_safe(pos_tensor):
    """
    Safely normalizes a [17, 3] COCO keypoint tensor using the Visibility flag.
    """
    coords = pos_tensor[:, :2].clone() # [17, 2]
    vis = pos_tensor[:, 2].clone()     # [17]
    
    valid_mask = vis > 0.0 
    
    if not valid_mask.any():
        return pos_tensor

    l_hip_valid = valid_mask[11].item()
    r_hip_valid = valid_mask[12].item()
    
    if l_hip_valid and r_hip_valid:
        root = (coords[11] + coords[12]) / 2.0
    elif l_hip_valid:
        root = coords[11] 
    elif r_hip_valid:
        root = coords[12] 
    else:
        l_sho_valid = valid_mask[5].item()
        r_sho_valid = valid_mask[6].item()
        if l_sho_valid and r_sho_valid:
            root = (coords[5] + coords[6]) / 2.0
        else:
            root = torch.tensor([0.0, 0.0], device=coords.device)

    coords[valid_mask] = coords[valid_mask] - root
    
    min_vals = coords[valid_mask].min(dim=0)[0]
    max_vals = coords[valid_mask].max(dim=0)[0]
    ranges = max_vals - min_vals
    global_scale = ranges.max()
    
    coords[valid_mask] = coords[valid_mask] / (global_scale + 1e-6)
    
    final_tensor = torch.cat([coords, vis.unsqueeze(1)], dim=1)
    return final_tensor

# Build input for model -- Normalize ipnut
def build_input(kp: np.ndarray, index_model: int, device):
    """
    Dynamically builds the input based on the selected model.
    kp : (17, 3) pixel coords (x, y, visibility)
    """
    # 1. Convert to tensor and apply the new 3-feature normalization
    kp_tensor = torch.tensor(kp, dtype=torch.float32)
    norm_tensor = normalize_coco_posture_safe(kp_tensor)
    
    if index_model == 0:
        # Flatten to [1, 34] for the Linear layers
        flat_tensor = norm_tensor.flatten().unsqueeze(0)
        return flat_tensor.to(device)
    
    elif index_model == 1:
        skeleton_edges = [
            [0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5,6], [5, 7], 
            [5, 11], [6, 12], [6, 8], [7, 9], [8, 10], [11, 12], [13, 11], 
            [14, 12], [15, 13], [16, 14]
        ]
        source, destination = [], []
        for u, v in skeleton_edges:
            source.extend([u, v])
            destination.extend([v, u])
            
        edge_index = torch.tensor([source, destination], dtype=torch.long)
        
        # Create PyG Data object
        data = Data(x=norm_tensor, edge_index=edge_index)
        return data.to(device)
    
    elif index_model == 2:
        cnn_input = norm_tensor.transpose(0, 1).unsqueeze(0)
        return cnn_input.to(device)

def prediction(model, data):
    """
    data: Either a flat Tensor (for MLP) or a PyG Data object (for GCN)
    """
    with torch.inference_mode():
        output = model(data)

    if output.shape[-1] == 1:
        prob = output.item()
        
        if prob < 0.0 or prob > 1.0:
            prob = torch.sigmoid(output).item()
            
        label = "Good" if prob >= 0.5 else "Bad"
        conf = prob if prob >= 0.5 else 1.0 - prob 

    else:
        # Convert raw logits to percentages (0 to 1)
        probs = torch.softmax(output, dim=-1).squeeze()
        
        # Find which class has the higher percentage
        pred_class = torch.argmax(probs).item()
        
        # NOTE: Assumes Class 0 is "Good". Change this if your dataset is flipped!
        label = "Good" if pred_class == 1 else "Bad"
        conf = probs[pred_class].item()

    return label, conf


############################ 
# Generated from Claude
############################

# Draw Skeleton on Frame
def draw_skeleton(frame, kp, label):
    #Colors
    GOOD = (50, 220, 100)
    BAD  = (60,  80, 230)
    KP   = (255, 210,  50)
    edge = GOOD if label == "Good" else (BAD if label == "Bad" else (120, 140, 180))

    for i, j in skeleton_edges:
        xi, yi, vi = kp[i]; xj, yj, vj = kp[j]
        if vi > 0.9 and vj > 0.9:
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)),
                     edge, 2, cv2.LINE_AA)

    for x, y, v in kp:
        if v > 0.9:
            cv2.circle(frame, (int(x), int(y)), 5, KP, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 0), 1, cv2.LINE_AA)
############################
# End
############################

# Sidebar 
with st.sidebar:
    st.markdown("## Sitting Posture Classification")
    st.divider()
    # !!!!!!!!!!!!!!! Add ur model path here
    model_list = [
        "mlp_latest_norm_best_model.pth",
        "gcn_model.pth",
        "1dcnn_best_model.pth"
    ]
    model_path = st.selectbox(
        "Change model here:",
        model_list,
        index=0
    )
    # 0 - MLP, 1 - GCN, 2 - 1DCNN, 3 - 
    index_model = model_list.index(model_path)

# Load models
model = posture_model = None
det_err = mdl_err = None

try:
    model = load_detector()
except Exception as e:
    det_err = str(e)

if model_path and os.path.exists(model_path):
    posture_model = load_posture_model(model_path, index_model)
else:
    mdl_err = f"File not found: `{model_path}`"

# Header
st.markdown("# Sitting Posture Classification")
st.caption("Real-time sitting posture · Keypoint R-CNN + GCN / 1DCNN / MLP / ?")

if det_err:
    st.error(f"**Keypoint R-CNN failed:** {det_err}")
    st.stop()

if mdl_err:
    st.warning(f"Posture model not loaded — {mdl_err}")

############################ 
# Generated from Claude
############################
# Body
c1, c2 = st.columns(2)
if c1.button("Start", width='stretch', type="primary"):
    st.session_state.update(running=True)
if c2.button("Stop", width='stretch'):
    st.session_state.running = False

col_vid, col_panel = st.columns([3, 1], gap="large")

with col_vid:
    banner     = st.empty()
    frame_slot = st.empty()
    banner.markdown(
        '<div class="status-idle"> Press Start to begin…</div>',
        unsafe_allow_html=True)

with col_panel:
    st.markdown("#### Live Info")
    card_conf = st.empty()
    st.divider()
    st.markdown("#### Keypoints (17)")
    kp_grid_slot = st.empty()

#Session state
for k, v in [("running", False), ("good", 0), ("bad", 0), ("total", 0)]:
    st.session_state.setdefault(k, v)

# Webcam 
if st.session_state.running:

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_n    = 0
    fps_window = []
    last_kps   = None
    last_label = None
    last_conf  = 0.0
    last_err   = None

    while st.session_state.running:
        t0 = time.time()

        ret, frame = cap.read()

        H, W = frame.shape[:2]
        frame_n += 1

        ############################ 
        # Coded by NJQ
        ############################
        # run R-CNN + classifier every 2 frames
        if frame_n % 2 == 0:
            last_err = None
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            kp = extract_keypoint(img, model, device)

            if kp is not None:
                last_kps = kp
                if posture_model is not None:
                    try:
                        input = build_input(kp, index_model, device)
                        input = input.to(device)
                        lbl, conf  = prediction(posture_model, input)
                        last_label = lbl
                        last_conf  = conf
                    except Exception as e:
                        last_err   = str(e)
                        last_label = None
            else:
                last_kps   = None
                last_label = None
        ############################ 
        # End
        ############################

        # Draw overlay
        vis = frame.copy()

        if last_kps is not None:
            draw_skeleton(vis, last_kps, last_label)
            for idx, (x, y, v) in enumerate(last_kps):
                    if v > 0.9:
                        cv2.putText(vis, parts[idx],
                                    (int(x) + 7, int(y) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (200, 210, 230), 1, cv2.LINE_AA)

        if last_label in ("Good", "Bad"):
            col_bgr = (50, 220, 100) if last_label == "Good" else (60, 80, 230)
            cv2.putText(vis, f"{last_label}  {last_conf*100:.0f}%",
                        (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3,
                        col_bgr, 2, cv2.LINE_AA)

        # FPS overlay
        elapsed = max(time.time() - t0, 1e-6)
        fps_window.append(1.0 / elapsed)
        if len(fps_window) > 30: fps_window.pop(0)
        fps = np.mean(fps_window)
        cv2.putText(vis, f"{fps:.0f} fps", (W - 110, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 70, 90), 1, cv2.LINE_AA)

        frame_slot.image(
            cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
            channels="RGB", width='stretch'
        )

        # Banner for status
        if last_err:
            banner.markdown(
                f'<div class="status-idle">Classifier error — see terminal for details</div>',
                unsafe_allow_html=True)
        elif last_label == "Good":
            banner.markdown(
                '<div class="status-good">Good Posture</div>',
                unsafe_allow_html=True)
        elif last_label == "Bad":
            banner.markdown(
                '<div class="status-bad">Bad Posture — Adjust your position!</div>',
                unsafe_allow_html=True)
        elif last_kps is None:
            banner.markdown(
                '<div class="status-idle">No person detected…</div>',
                unsafe_allow_html=True)
        else:
            banner.markdown(
                '<div class="status-idle">Classifying…</div>',
                unsafe_allow_html=True)

        # Info cards
        pct     = last_conf * 100
        bar_col = ("#22c55e" if last_label == "Good"
                   else "#ef4444" if last_label == "Bad"
                   else "#2d3748")

        card_conf.markdown(f"""
            <div class="card">
            <div class="card-label">Confidence</div>
            <div class="card-value">{pct:.1f}%</div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{pct:.0f}%;background:{bar_col};"></div>
            </div>
            </div>""", unsafe_allow_html=True
        )

        # Keypoint chip grid
        if last_kps is not None:
            chips = "".join(
                f'<div class="kp-chip{"  active" if v > 0.9 else ""}">'
                f'{parts[idx]}<br>'
                f'<span style="color:#374151">{x:.0f},{y:.0f}</span></div>'
                for idx, (x, y, v) in enumerate(last_kps)
            )
            kp_grid_slot.markdown(
                f'<div class="kp-grid">{chips}</div>',
                unsafe_allow_html=True)

        time.sleep(0.005)

    cap.release()
    banner.markdown(
        '<div class="status-idle">Stopped.</div>',
        unsafe_allow_html=True)
############################ 
# End
############################
