import torch
import cv2
import numpy as np
import base64
import os
import gc
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
from supabase import create_client
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

# 1. Patch torch
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
original_torch_load = torch.load
torch.load = patched_torch_load

# 2. Khởi tạo FastAPI (PHẢI NẰM TRƯỚC CÁC ROUTE)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 3. Khởi tạo Supabase
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# 4. Biến toàn cục
yolo_pose = None
fas_predict = None
image_cropper = None

# 5. Hàm nạp model (Lazy Loading)
def load_models():
    global yolo_pose, fas_predict, image_cropper
    if yolo_pose is None:
        print("[INFO] Đang nạp model...")
        yolo_pose = YOLO('models/best.pt')
        yolo_pose.model.float()
        fas_predict = AntiSpoofPredict(device_id=-1)
        image_cropper = CropImage()
        print("[INFO] Nạp model hoàn tất!")

# 6. CÁC ROUTE (PHẢI NẰM SAU app = FastAPI())
@app.get("/health")
async def health():
    return {"status": "ok"}

class ImageRequest(BaseModel):
    image: str

@app.post("/recognize")
async def recognize(request: ImageRequest):
    load_models()
    try:
        # Decode ảnh
        img_data = base64.b64decode(request.image.split(',')[-1] if ',' in request.image else request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"status": "error", "message": "cv2 decode failed"}

        # Logic YOLO
        results = yolo_pose(frame, verbose=False)
        for result in results:
            if len(result.boxes) > 0:
                box = result.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                return {"status": "success", "detected": True, "box": [x1, y1, x2, y2]}
        
        return {"status": "success", "detected": False, "message": "Không thấy mặt"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
