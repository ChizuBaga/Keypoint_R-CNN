import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import time
import os
import threading
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from PIL import Image, ImageTk

# GCN Specific imports
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import joblib

# CUDA / CPU
device = "cpu"

#############################
# Coded by Dennis
#############################

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

#############################
# End
#############################

# --- Models Definitions ---
def load_detector():
    model  = keypointrcnn_resnet50_fpn(weights='DEFAULT')
    model.to(device).eval()
    return model
###########################
# Code by Ng Jia Qin
###########################
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
###########################
# End
###########################

###########################
# Code by Dennis
###########################
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
###########################
# End
###########################

###########################
# Code by Pang Jing Thean
###########################
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
###########################
# End
###########################

def load_posture_model(path: str, index_model: int):

    if path.endswith('.pkl'):
        model = joblib.load(path)
        return model
        
    # Load PyTorch models
    if index_model == 0:
        net = MLP()
    elif index_model == 1:
        net = GCN_model()
    elif index_model == 2:
        net = CNN1d()
    
    net.load_state_dict(torch.load(path, map_location=device))
    net.to(device).eval()
    return net

def extract_keypoint(img, model, device):
    img_for_drawing = F.to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_for_drawing)[0]

    if len(output["scores"]) == 0: 
        return None
        
    best = int(torch.argmax(output["scores"]).item()) 
    if output["scores"][best] < 0.9: 
        return None 

    kp = output["keypoints"][best].clone()
    kp_scores = output["keypoints_scores"][best]
    for i in range(17):
        if kp_scores[i] < 0.0:
            kp[i, :] = torch.tensor([0.0, 0.0, 0.0], device=kp.device)

    return kp.detach().cpu().numpy()

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
    elif index_model == 3:
        flat_array = norm_tensor.flatten().numpy()
        return flat_array.reshape(1, -1)
    
###########################
# Generated by Gen AI
###########################

def prediction(model, data):

    if hasattr(model, 'predict'):
        pred_class = model.predict(data)[0]
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(data)[0]
            conf = float(np.max(probs))
        else:
            conf = None
            
        label = "Good" if pred_class == 1 else "Bad"
        return label, conf

    with torch.inference_mode():
        output = model(data)
        
    if output.shape[-1] == 1: 
        prob = torch.sigmoid(output).item()
        
        label = "Good" if prob >= 0.5 else "Bad"
        conf = prob if prob >= 0.5 else 1.0 - prob 
    else:
        # Fallback for multi-class (if you ever add one)
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
        self.root.title("Sitting Posture Classification - True Async")
        self.root.geometry("1100x700")

        # Threading & State Controls
        self.running = False
        self.cap = None
        self.data_lock = threading.Lock()
        
        # Shared Data Buffers
        self.raw_frame = None       
        self.latest_kps = None
        self.is_new_kp = False  
        self.last_label = None
        self.last_conf = 0.0
        self.last_err = None
        self.inference_latency = 0.0 

        # --- NEW: Benchmarking Metrics ---
        self.inference_count = 0
        self.total_latency = 0.0
        self.total_confidence = 0.0

        self.last_ui_time = time.time()
        self.fps_window = []

        self.model_list = [
            "MLP.pth",
            "gcn_model.pth",
            "1D-CNN.pth",
            "best_svc_posture_model.pkl"
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

        self.lbl_latency = ttk.Label(self.right_frame, text="Latency: 0.0 ms", font=("Courier", 12, "bold"), foreground="blue")
        self.lbl_latency.pack(anchor=tk.W, pady=5)

        # --- NEW: Benchmarking UI Elements ---
        ttk.Separator(self.right_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(self.right_frame, text="Averages (Current Model)", font=("Arial", 10, "italic")).pack(anchor=tk.W)

        self.lbl_avg_conf = ttk.Label(self.right_frame, text="Avg Conf: 0.0%", font=("Courier", 11))
        self.lbl_avg_conf.pack(anchor=tk.W, pady=2)

        self.lbl_avg_latency = ttk.Label(self.right_frame, text="Avg Latency: 0.00 ms", font=("Courier", 11))
        self.lbl_avg_latency.pack(anchor=tk.W, pady=2)

        self.lbl_inf_count = ttk.Label(self.right_frame, text="Total Inferences: 0", font=("Courier", 11))
        self.lbl_inf_count.pack(anchor=tk.W, pady=2)

        ttk.Separator(self.right_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        ttk.Label(self.right_frame, text="Keypoints Found:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(5, 5))
        self.txt_kps = tk.Text(self.right_frame, height=15, width=30, bg="#f0f0f0", state=tk.DISABLED)
        self.txt_kps.pack(fill=tk.BOTH, expand=True)

    def load_selected_model(self, event=None):
        model_name = self.cb_model.get()
        with self.data_lock:

            self.inference_count = 0
            self.total_latency = 0.0
            self.total_confidence = 0.0
            self.last_conf = 0.0
            self.inference_latency = 0.0
            
            self.index_model = self.model_list.index(model_name)
            if os.path.exists(model_name):
                try:
                    self.posture_model = load_posture_model(model_name, self.index_model)
                    print(f"Loaded {model_name} successfully.")
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    self.posture_model = None
            else:
                print(f"Warning: Model file {model_name} not found.")
                self.posture_model = None

    def start_video(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.running = True
            
            # Start background workers
            threading.Thread(target=self.camera_worker, daemon=True).start()
            threading.Thread(target=self.rcnn_worker, daemon=True).start()
            threading.Thread(target=self.inference_worker, daemon=True).start()
            
            self.refresh_ui_loop()

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.lbl_banner.config(text="Stopped.", bg="#1e2535", fg="#4b5568")
        self.video_panel.config(image="")

    # --- THREAD 1: Pure Camera Capture (Super Fast) ---
    def camera_worker(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.data_lock:
                    self.raw_frame = frame
            else:
                time.sleep(0.01)

    # --- THREAD 2: Keypoint RCNN (Runs as fast as hardware allows) ---
    def rcnn_worker(self):
        while self.running:
            with self.data_lock:
                frame = self.raw_frame.copy() if self.raw_frame is not None else None

            if frame is not None:
                orig_h, orig_w = frame.shape[:2]
                
                small_w, small_h = 640, 360 
                small_frame = cv2.resize(frame, (small_w, small_h))
                img = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                kp = extract_keypoint(img, self.detector, device)
                
                if kp is not None:
                    scale_x = orig_w / small_w
                    scale_y = orig_h / small_h
                    kp[:, 0] *= scale_x
                    kp[:, 1] *= scale_y
                
                with self.data_lock:
                    self.latest_kps = kp
                    self.is_new_kp = True
            else:
                time.sleep(0.05)

    # --- THREAD 3: Posture Classification (Every 2 Seconds) ---
    def inference_worker(self):
        while self.running:
            time.sleep(2.0)
            
            with self.data_lock:
                kps_to_infer = self.latest_kps
                model = self.posture_model
                idx = self.index_model
                is_fresh = self.is_new_kp

            if kps_to_infer is not None and model is not None and is_fresh:
                try:
                    start_time = time.perf_counter() 
                    
                    inp = build_input(kps_to_infer, idx, device)
                    lbl, conf = prediction(model, inp)
                    
                    end_time = time.perf_counter()
                    latency_ms = (end_time - start_time) * 1000

                    with self.data_lock:
                        self.last_label = lbl
                        self.last_conf = conf
                        self.inference_latency = latency_ms
                        self.last_err = None
                        
                        self.inference_count += 1
                        self.total_latency += latency_ms

                        if conf is not None:
                            self.total_confidence += conf
                        self.is_new_kp = False
                        
                except Exception as e:
                    with self.data_lock:
                        self.last_err = str(e)
                        self.last_label = None

    # --- MAIN THREAD: UI & Overlay Logic ---
    def refresh_ui_loop(self):
        if not self.running: 
            return

        with self.data_lock:
            frame_to_show = self.raw_frame.copy() if self.raw_frame is not None else None
            current_kps = self.latest_kps
            lbl = self.last_label
            conf = self.last_conf
            err = self.last_err
            latency = self.inference_latency
            
            inf_count = self.inference_count
            tot_lat = self.total_latency
            tot_conf = self.total_confidence

        if frame_to_show is not None:

            if current_kps is not None:
                draw_skeleton(frame_to_show, current_kps, lbl)
                for idx, (x, y, v) in enumerate(current_kps):
                    if v > 0.9:
                        cv2.putText(frame_to_show, parts[idx], (int(x) + 7, int(y) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 230), 1, cv2.LINE_AA)

            if lbl in ("Good", "Bad"):
                col_bgr = (50, 220, 100) if lbl == "Good" else (60, 80, 230)
                
                conf_text = f"  {conf*100:.0f}%" if conf is not None else "  (SVC)"
                cv2.putText(frame_to_show, f"{lbl}{conf_text}",
                            (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, col_bgr, 2, cv2.LINE_AA)

            # Calculate actual UI FPS
            current_time = time.time()
            elapsed = max(current_time - self.last_ui_time, 1e-6)
            self.last_ui_time = current_time
            self.fps_window.append(1.0 / elapsed)
            if len(self.fps_window) > 30: self.fps_window.pop(0)
            
            fps = np.mean(self.fps_window)
            H, W = frame_to_show.shape[:2]
            cv2.putText(frame_to_show, f"Webcam FPS: {fps:.0f}", (W - 180, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 70, 90), 1, cv2.LINE_AA)

            # Render to Tkinter
            vis_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(vis_rgb).resize((800, 450), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_panel.imgtk = imgtk
            self.video_panel.configure(image=imgtk)

        if err:
            self.lbl_banner.config(text="Classifier error — see terminal", bg="#1e2535", fg="#ef4444")
        elif lbl == "Good":
            self.lbl_banner.config(text="GOOD POSTURE", bg="#22c55e", fg="white")
        elif lbl == "Bad":
            self.lbl_banner.config(text="BAD POSTURE — Adjust!", bg="#ef4444", fg="white")
        elif current_kps is None:
            self.lbl_banner.config(text="Waiting for person...", bg="#1e2535", fg="#4b5568")
        else:
            self.lbl_banner.config(text="Classifying...", bg="#1e2535", fg="#4b5568")

        if conf is not None:
            self.lbl_conf.config(text=f"Confidence: {conf*100:.1f}%")
        else:
            self.lbl_conf.config(text="Confidence: N/A")
            
        self.lbl_latency.config(text=f"Latency: {latency:.2f} ms")

        avg_lat = (tot_lat / inf_count) if inf_count > 0 else 0.0
        
        if conf is not None:
            avg_conf = (tot_conf / inf_count) if inf_count > 0 else 0.0
            self.lbl_avg_conf.config(text=f"Avg Conf: {avg_conf*100:.1f}%")
        else:
            self.lbl_avg_conf.config(text="Avg Conf: N/A")
            
        self.lbl_avg_latency.config(text=f"Avg Latency: {avg_lat:.2f} ms")
        self.lbl_inf_count.config(text=f"Total Inferences: {inf_count}")

        # Update Keypoints text
        self.txt_kps.config(state=tk.NORMAL)
        self.txt_kps.delete(1.0, tk.END)
        if current_kps is not None:
            kps_text = ""
            for idx, (x, y, v) in enumerate(current_kps):
                status = "✓" if v > 0.9 else "x"
                kps_text += f"[{status}] {parts[idx]}: {int(x)}, {int(y)}\n"
            self.txt_kps.insert(tk.END, kps_text)
        self.txt_kps.config(state=tk.DISABLED)

        # Refresh UI as fast as possible (~30ms = ~30 FPS)
        self.root.after(30, self.refresh_ui_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()

############################
# END
###########################