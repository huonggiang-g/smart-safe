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

# Patch để tránh lỗi weights_only
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
original_torch_load = torch.load
torch.load = patched_torch_load

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Khởi tạo Supabase
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# NẠP MODEL NGAY LÚC KHỞI ĐỘNG (Tận dụng RAM của Standard Tier)
print("[INFO] Đang nạp model vào RAM...")
yolo_pose = YOLO('models/best.pt')
yolo_pose.model.float() 
print("[INFO] Nạp model hoàn tất!")

class ImageRequest(BaseModel):
    image: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/recognize")
async def recognize(request: ImageRequest):
    try:
        # Decode ảnh
        img_data = base64.b64decode(request.image.split(',')[-1] if ',' in request.image else request.image)
        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # YOLO nhận diện
        results = yolo_pose(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                
                # DeepFace (VGG-Face)
                res = DeepFace.represent(img_path=face_crop, model_name="VGG-Face", detector_backend="skip", enforce_detection=False)
                current_vec = np.array(res[0]["embedding"])
                
                # Supabase so sánh
                response = supabase.table("faces").select("name, embedding").execute()
                best_name, min_dist = "Người lạ", 0.4
                for item in response.data:
                    dist = cosine(current_vec, np.array(item["embedding"]))
                    if dist < min_dist:
                        min_dist, best_name = dist, item["name"]
                
                return {"recognized": best_name != "Người lạ", "name": best_name}
        
        return {"recognized": False, "message": "No face"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
