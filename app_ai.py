import torch
# Patch để ép PyTorch chấp nhận load model cũ
import sys
if hasattr(torch, 'serialization'):
    torch.serialization.add_safe_globals([dict])
import cv2
import numpy as np
import base64
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
from supabase import create_client
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
import warnings

# 1. Khởi tạo Supabase
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# 2. Định nghĩa hằng số
MODEL_FAS = 'models/2.7_80x80_MiniFASNetV2.pth'
THRESHOLD_FACENET = 0.35

# 3. Khởi tạo mô hình
print("[INFO] Đang nạp model...")
yolo_pose = YOLO('models/best.pt')
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()
print("[INFO] Khởi tạo hoàn tất!")

class ImageRequest(BaseModel):
    image: str

@app.post("/recognize")
async def recognize(request: ImageRequest):
    try:
        # Decode ảnh
        img_data = base64.b64decode(request.image.split(',')[-1] if ',' in request.image else request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Detect mặt
        results = yolo_pose(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # FAS (Chống giả mạo)
                img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
                pred = fas_predict.predict(img_crop_fas, MODEL_FAS)
                
                if float(pred[0][1]) < 0.85:
                    return {"recognized": False, "detected": True, "error": "Spoof detected"}

                # Nhận diện (DeepFace) - PHẢI NẰM TRONG VÒNG LẶP ĐỂ CÓ face_crop
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                res = DeepFace.represent(img_path=face_crop, model_name="Facenet512", detector_backend="skip", enforce_detection=False)
                current_vec = np.array(res[0]["embedding"])
                
                # Truy vấn Supabase
                response = supabase.table("faces").select("name, embedding").execute()
                
                # So sánh cosine
                best_name, min_dist = "Unknown", 1.0
                for item in response.data:
                    v = np.array(item["embedding"])
                    dist = cosine(current_vec, v)
                    if dist < THRESHOLD_FACENET and dist < min_dist:
                        min_dist, best_name = dist, item["name"]
                
                return {"recognized": True if best_name != "Unknown" else False, "name": best_name}
        
        return {"recognized": False, "detected": False, "message": "No face detected"}

    except Exception as e:
        return {"error": str(e)}
