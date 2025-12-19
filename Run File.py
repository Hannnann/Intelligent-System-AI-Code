# =============================================================================
# DERMNET SKIN DISEASE DETECTION — FASTAPI + NGROK (COLAB SAFE)
# =============================================================================

# -----------------------------
# INSTALL DEPENDENCIES
# -----------------------------
!pip install -q fastapi uvicorn python-multipart pyngrok timm pillow torch torchvision nest_asyncio

# -----------------------------
# IMPORTS
# -----------------------------
import os
import io
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict

import torchvision.transforms as T
import timm

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import uvicorn
import nest_asyncio

# -----------------------------
# CONFIGURATION
# -----------------------------
EXPORT_DIR = "./runs/dermnet_agents_20251217_123433/export"
DEVICE = "cpu"
TOP_K = 3

NGROK_AUTHTOKEN = "36yVIO1x0jDCVn0CCFfTYFqQ4Ph_5Z6bo1MhyjEQCntq2a77K"  # 🔴 MUST BE REAL

# -----------------------------
# NGROK AUTH
# -----------------------------
ngrok.set_auth_token(NGROK_AUTHTOKEN)
d
# -----------------------------
# UTILS
# -----------------------------
def softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

# =============================================================================
# INFERENCE HELPER (SELF CONTAINED)
# =============================================================================
class InferenceHelper:
    def __init__(self, export_dir: str, device: str = "cpu"):
        self.device = device

        # -------------------------
        # Load manifest
        # -------------------------
        with open(os.path.join(export_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)

        self.img_size = int(self.manifest["img_size"])
        self.conf_threshold = float(self.manifest["confidence_threshold"])

        # -------------------------
        # Load labels
        # -------------------------
        with open(os.path.join(export_dir, "labels.json"), "r") as f:
            labels = json.load(f)

        self.idx2label = {int(k): v for k, v in labels["idx2label"].items()}
        self.num_classes = len(self.idx2label)

        # -------------------------
        # Load model
        # -------------------------
        self.model = timm.create_model(
            self.manifest["model_name"],
            pretrained=False,
            num_classes=self.num_classes
        )

        ckpt = torch.load(
            os.path.join(export_dir, "model_best.pt"),
            map_location=device
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(device)
        self.model.eval()

        # -------------------------
        # Load temperature scaler
        # -------------------------
        self.temperature = 1.0
        tpath = os.path.join(export_dir, "temperature_scaler.pt")
        if os.path.exists(tpath):
            temp_ckpt = torch.load(tpath, map_location="cpu")
            self.temperature = float(temp_ckpt.get("temperature", 1.0))

        # -------------------------
        # Image transforms
        # -------------------------
        self.tf = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        print("✅ MODEL LOADED")
        print("Classes:", self.idx2label)
        print("Temperature:", self.temperature)

    @torch.no_grad()
    def predict_bytes(self, img_bytes: bytes, topk: int = 3) -> Dict:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)

        logits = self.model(x).cpu().numpy()
        logits = logits / max(self.temperature, 1e-6)

        probs = softmax_np(logits)[0]
        order = np.argsort(-probs)[:topk]

        preds = []
        for idx in order:
            preds.append({
                "label": self.idx2label[int(idx)],
                "confidence": float(probs[int(idx)])
            })

        top_conf = preds[0]["confidence"]
        uncertain = top_conf < self.conf_threshold

        return {
            "predictions": preds,
            "top_confidence": top_conf,
            "uncertain": uncertain,
            "disclaimer": "AI suggestion only. Not a medical diagnosis."
        }

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="Skin Disease Detection API",
    description="DermNet-based AI",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD MODEL
# -----------------------------
helper = InferenceHelper(EXPORT_DIR, DEVICE)

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def health():
    return {
        "status": "running",
        "model": "DermNet EfficientNet",
        "classes": list(helper.idx2label.values())
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img_bytes = await image.read()
    return helper.predict_bytes(img_bytes, TOP_K)

# =============================================================================
# START SERVER (COLAB SAFE)
# =============================================================================
nest_asyncio.apply()

public_url = ngrok.connect(8000)
print("🚀 PUBLIC API URL:", public_url)

await uvicorn.Server(
    uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
).serve()
