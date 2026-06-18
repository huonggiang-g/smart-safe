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

# Patch torch
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
original_torch_load = torch.load
torch.load = patched_torch_load

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# Biến toàn cục
yolo_pose = None

def load_models():
    global yolo_pose
    if yolo_pose is None:
        print("[INFO] Đang nạp YOLO...")
        yolo_pose = YOLO('models/best.pt')
        yolo_pose.model.float()
        print("[INFO] Nạp YOLO hoàn tất!")

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
        if frame is None: return {"status": "error", "message": "cv2 decode failed"}

        # 1. YOLO: Tìm mặt
        results = yolo_pose(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 2. VGG-Face: Định danh (không FAS)
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                
                # DeepFace nạp model khi gọi, nên nếu RAM thiếu, nó sẽ báo ngay tại đây
                res = DeepFace.represent(img_path=face_crop, model_name="VGG-Face", detector_backend="skip", enforce_detection=False)
                current_vec = np.array(res[0]["embedding"])
                
                # 3. Supabase: So sánh
                response = supabase.table("faces").select("name, embedding").execute()
                best_name, min_dist = "Người lạ", 0.4 
                for item in response.data:
                    dist = cosine(current_vec, np.array(item["embedding"]))
                    if dist < min_dist:
                        min_dist, best_name = dist, item["name"]
                
                # Dọn dẹp RAM cực kỳ quan trọng
                del res, current_vec
                gc.collect()
                
                return {
                    "recognized": True if best_name != "Người lạ" else False, 
                    "name": best_name, 
                    "confidence": float(1 - min_dist)
                }
        
        return {"recognized": False, "message": "Không thấy mặt"}
    except Exception as e:
        print(f"LỖI: {str(e)}")
        return {"status": "error", "message": str(e)}
