import torch
import cv2
import numpy as np
import base64
import gc
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
from supabase import create_client
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Khởi tạo Supabase
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# Nạp model lúc khởi động (Dùng Facenet512 để nhẹ RAM)
print("[INFO] Nạp model bắt đầu...")
yolo_pose = YOLO('models/best.pt')
yolo_pose.model.float()
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()
print("[INFO] Hệ thống sẵn sàng!")

class ImageRequest(BaseModel):
    image: str

@app.post("/recognize")
async def recognize(request: ImageRequest):
    try:
        # Decode ảnh
        img_data = base64.b64decode(request.image.split(',')[-1])
        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        # 1. YOLO nhận diện
        results = yolo_pose(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 2. FAS chống giả mạo
                img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
                pred = fas_predict.predict(img_crop_fas, 'models/2.7_80x80_MiniFASNetV2.pth')
                if float(pred[0][1]) < 0.85:
                    return {"recognized": False, "message": "Spoof detected"}

                # 3. Facenet512 nhận diện danh tính
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                res = DeepFace.represent(img_path=face_crop, model_name="Facenet512", detector_backend="skip", enforce_detection=False)
                current_vec = np.array(res[0]["embedding"])
                
                # 4. So sánh Supabase
                response = supabase.table("faces").select("name, embedding").execute()
                best_name, min_dist = "Người lạ", 0.4
                for item in response.data:
                    dist = cosine(current_vec, np.array(item["embedding"]))
                    if dist < min_dist:
                        min_dist, best_name = dist, item["name"]
                
                # Dọn dẹp RAM
                gc.collect()
                return {"recognized": best_name != "Người lạ", "name": best_name}
        
        return {"recognized": False, "message": "No face"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
