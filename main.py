import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from PIL import Image, ImageTk

# GCN Specific imports
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# CUDA / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# --- Models Definitions ---
def load_detector():
    model  = keypointrcnn_resnet50_fpn(weights='DEFAULT') # Updated from pretrained=True
    model.to(device).eval()
    return model

class MLP(nn.Module):
    def __init__(self):
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.outputlayer(x)
        x = self.sigmoid(x)
        return x

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
        x = x.view(-1, 17 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

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

def load_posture_model(path: str, index_model: int):
    if index_model == 0:
        net = MLP()
    elif index_model == 1:
        net = GCN_model()
    elif index_model == 2:
        net = CNN1d()
    
    net.load_state_dict(torch.load(path, map_location=device))
    net.to(device).eval()
    return net

# --- Helper Functions ---
def extract_keypoint(img, model, device):
    img_for_drawing = F.to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_for_drawing)[0]

    kp = output["keypoints"]
    scores = output["scores"].cpu().numpy()
    if len(scores) == 0: return None
    best = int(np.argmax(scores)) 
    if scores[best] < 0.9: return None 
    return kp[best].detach().cpu().numpy()

def normalize_coco_posture_safe(pos_tensor):
    coords = pos_tensor[:, :2].clone() 
    vis = pos_tensor[:, 2].clone()     
    valid_mask = vis > 0.0 
    
    if not valid_mask.any(): return pos_tensor

    l_hip_valid, r_hip_valid = valid_mask[11].item(), valid_mask[12].item()
    if l_hip_valid and r_hip_valid: root = (coords[11] + coords[12]) / 2.0
    elif l_hip_valid: root = coords[11] 
    elif r_hip_valid: root = coords[12] 
    else:
        l_sho_valid, r_sho_valid = valid_mask[5].item(), valid_mask[6].item()
        if l_sho_valid and r_sho_valid: root = (coords[5] + coords[6]) / 2.0
        else: root = torch.tensor([0.0, 0.0], device=coords.device)

    coords[valid_mask] = coords[valid_mask] - root
    min_vals = coords[valid_mask].min(dim=0)[0]
    max_vals = coords[valid_mask].max(dim=0)[0]
    ranges = max_vals - min_vals
    global_scale = ranges.max()
    coords[valid_mask] = coords[valid_mask] / (global_scale + 1e-6)
    
    return torch.cat([coords, vis.unsqueeze(1)], dim=1)

def build_input(kp: np.ndarray, index_model: int, device):
    kp_tensor = torch.tensor(kp, dtype=torch.float32)
    norm_tensor = normalize_coco_posture_safe(kp_tensor)
    
    if index_model == 0:
        flat_tensor = norm_tensor.flatten().unsqueeze(0)
        return flat_tensor.to(device)
    elif index_model == 1:
        source, destination = [], []
        for u, v in skeleton_edges:
            source.extend([u, v])
            destination.extend([v, u])
        edge_index = torch.tensor([source, destination], dtype=torch.long)
        data = Data(x=norm_tensor, edge_index=edge_index)
        return data.to(device)
    elif index_model == 2:
        cnn_input = norm_tensor.transpose(0, 1).unsqueeze(0)
        return cnn_input.to(device)

def prediction(model, data):
    with torch.inference_mode():
        output = model(data)
    if output.shape[-1] == 1: 
        prob = output.item()
        if prob < 0.0 or prob > 1.0: prob = torch.sigmoid(output).item()
        label = "Good" if prob >= 0.5 else "Bad"
        conf = prob if prob >= 0.5 else 1.0 - prob 
    else:
        probs = torch.softmax(output, dim=-1).squeeze()
        pred_class = torch.argmax(probs).item()
        label = "Good" if pred_class == 1 else "Bad"
        conf = probs[pred_class].item()
    return label, conf

def draw_skeleton(frame, kp, label):
    GOOD, BAD, KP = (50, 220, 100), (60, 80, 230), (255, 210, 50)
    edge = GOOD if label == "Good" else (BAD if label == "Bad" else (120, 140, 180))

    for i, j in skeleton_edges:
        xi, yi, vi = kp[i]; xj, yj, vj = kp[j]
        if vi > 0.9 and vj > 0.9:
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), edge, 2, cv2.LINE_AA)

    for x, y, v in kp:
        if v > 0.9:
            cv2.circle(frame, (int(x), int(y)), 5, KP, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 0), 1, cv2.LINE_AA)


# --- Tkinter Application Class ---
class PostureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sitting Posture Classification")
        self.root.geometry("1100x700")

        self.running = False
        self.cap = None
        self.frame_n = 0
        self.fps_window = []
        
        self.last_kps = None
        self.last_label = None
        self.last_conf = 0.0
        self.last_err = None

        # Timer to track last inference
        self.last_inference_time = 0

        self.model_list = [
            "mlp_latest_norm_best_model.pth",
            "gcn_model.pth",
            "1dcnn_best_model.pth"
        ]
        self.index_model = 0
        self.posture_model = None
        
        print("Loading Keypoint R-CNN...")
        self.detector = load_detector()
        print("Detector loaded.")

        self.setup_ui()
        self.load_selected_model()

    def setup_ui(self):
        # Main frames
        self.left_frame = ttk.Frame(self.root, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root, width=300, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Video Banner & Feed
        self.lbl_banner = tk.Label(self.left_frame, text="Press Start to begin...", font=("Arial", 16, "bold"), bg="#1e2535", fg="#4b5568", pady=10)
        self.lbl_banner.pack(fill=tk.X, pady=(0, 10))

        self.video_panel = tk.Label(self.left_frame, bg="black")
        self.video_panel.pack(fill=tk.BOTH, expand=True)

        # Sidebar Controls
        ttk.Label(self.right_frame, text="Settings", font=("Arial", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(self.right_frame, text="Select Model:").pack(anchor=tk.W)
        self.cb_model = ttk.Combobox(self.right_frame, values=self.model_list, state="readonly")
        self.cb_model.current(0)
        self.cb_model.pack(fill=tk.X, pady=(0, 20))
        self.cb_model.bind("<<ComboboxSelected>>", self.load_selected_model)

        # Buttons
        btn_frame = ttk.Frame(self.right_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        self.btn_start = ttk.Button(btn_frame, text="Start", command=self.start_video)
        self.btn_start.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.btn_stop = ttk.Button(btn_frame, text="Stop", command=self.stop_video)
        self.btn_stop.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

        # Stats
        ttk.Separator(self.right_frame, orient='horizontal').pack(fill=tk.X, pady=20)
        ttk.Label(self.right_frame, text="Live Info", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.lbl_conf = ttk.Label(self.right_frame, text="Confidence: 0.0%", font=("Courier", 12))
        self.lbl_conf.pack(anchor=tk.W, pady=5)

        ttk.Label(self.right_frame, text="Keypoints Found:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(15, 5))
        self.txt_kps = tk.Text(self.right_frame, height=15, width=30, bg="#f0f0f0", state=tk.DISABLED)
        self.txt_kps.pack(fill=tk.BOTH, expand=True)

    def load_selected_model(self, event=None):
        model_name = self.cb_model.get()
        self.index_model = self.model_list.index(model_name)
        if os.path.exists(model_name):
            try:
                self.posture_model = load_posture_model(model_name, self.index_model)
                print(f"Loaded {model_name} successfully.")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                self.posture_model = None
        else:
            print(f"Warning: Model file {model_name} not found in directory.")
            self.posture_model = None

    def start_video(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.running = True
            self.frame_n = 0
            self.update_frame()

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.lbl_banner.config(text="Stopped.", bg="#1e2535", fg="#4b5568")
        self.video_panel.config(image="")

    def update_frame(self):
        if not self.running: return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        t0 = time.time()
        H, W = frame.shape[:2]
        self.frame_n += 1

        # Run inference every 10s
        current_time = time.time()

        if current_time - self.last_inference_time >= 10.0:
            self.last_err = None
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run the heavy models
            kp = extract_keypoint(img, self.detector, device)

            if kp is not None:
                self.last_kps = kp
                if self.posture_model is not None:
                    try:
                        inp = build_input(kp, self.index_model, device).to(device)
                        lbl, conf = prediction(self.posture_model, inp)
                        self.last_label = lbl
                        self.last_conf = conf
                    except Exception as e:
                        self.last_err = str(e)
                        self.last_label = None
            else:
                self.last_kps = None
                self.last_label = None
                
            # Reset the timer
            self.last_inference_time = current_time

        vis = frame.copy()

        # Draw overlays
        if self.last_kps is not None:
            draw_skeleton(vis, self.last_kps, self.last_label)
            for idx, (x, y, v) in enumerate(self.last_kps):
                if v > 0.9:
                    cv2.putText(vis, parts[idx], (int(x) + 7, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 230), 1, cv2.LINE_AA)

        if self.last_label in ("Good", "Bad"):
            col_bgr = (50, 220, 100) if self.last_label == "Good" else (60, 80, 230)
            cv2.putText(vis, f"{self.last_label}  {self.last_conf*100:.0f}%",
                        (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, col_bgr, 2, cv2.LINE_AA)

        # FPS calculation
        elapsed = max(time.time() - t0, 1e-6)
        self.fps_window.append(1.0 / elapsed)
        if len(self.fps_window) > 30: self.fps_window.pop(0)
        fps = np.mean(self.fps_window)
        cv2.putText(vis, f"{fps:.0f} fps", (W - 110, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 70, 90), 1, cv2.LINE_AA)

        # Update UI Elements
        self.update_ui_state()

        # Convert OpenCV image for Tkinter
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(vis_rgb)
        
        # Resize to fit panel dynamically or fix size (using a fixed size for simplicity)
        img_pil = img_pil.resize((800, 450), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        
        self.video_panel.imgtk = imgtk
        self.video_panel.configure(image=imgtk)

        # Schedule next frame
        self.root.after(10, self.update_frame)

    def update_ui_state(self):
        # Status Banner
        if self.last_err:
            self.lbl_banner.config(text="Classifier error — see terminal", bg="#1e2535", fg="#ef4444")
        elif self.last_label == "Good":
            self.lbl_banner.config(text="GOOD POSTURE", bg="#22c55e", fg="white")
        elif self.last_label == "Bad":
            self.lbl_banner.config(text="BAD POSTURE — Adjust!", bg="#ef4444", fg="white")
        elif self.last_kps is None:
            self.lbl_banner.config(text="No person detected...", bg="#1e2535", fg="#4b5568")
        else:
            self.lbl_banner.config(text="Classifying...", bg="#1e2535", fg="#4b5568")

        # Confidence
        self.lbl_conf.config(text=f"Confidence: {self.last_conf*100:.1f}%")

        # Keypoints Text Box
        self.txt_kps.config(state=tk.NORMAL)
        self.txt_kps.delete(1.0, tk.END)
        if self.last_kps is not None:
            kps_text = ""
            for idx, (x, y, v) in enumerate(self.last_kps):
                status = "✓" if v > 0.9 else "x"
                kps_text += f"[{status}] {parts[idx]}: {int(x)}, {int(y)}\n"
            self.txt_kps.insert(tk.END, kps_text)
        self.txt_kps.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()