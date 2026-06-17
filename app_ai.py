import torch
import sys

# MỞ KHÓA TOÀN CỤC: Ép torch.load mặc định luôn luôn là weights_only=False
# Điều này vô hiệu hóa hoàn toàn chính sách bảo mật mới của PyTorch 2.6
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

original_torch_load = torch.load
torch.load = patched_torch_load
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

# Khởi tạo FastAPI
app = FastAPI()
print("DEBUG: app đã được khởi tạo thành công!")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Hoặc thay bằng URL của trang web bạn
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối Supabase
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# Định nghĩa hằng số
MODEL_FAS = 'models/2.7_80x80_MiniFASNetV2.pth'
THRESHOLD_FACENET = 0.35

# Khởi tạo mô hình
print("[INFO] Đang nạp model...")
yolo_pose = YOLO('models/best.pt')
yolo_pose.model.float() 
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()
print("[INFO] Khởi tạo hoàn tất!")

class ImageRequest(BaseModel):
    image: str

@app.post("/recognize")
async def recognize(request: ImageRequest):
    try:
        print(f"DEBUG: Nhận request mới. Độ dài base64: {len(request.image)}")
        # Decode ảnh
        img_data = base64.b64decode(request.image.split(',')[-1] if ',' in request.image else request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("ERROR: cv2.imdecode trả về None! Ảnh bị hỏng hoặc sai định dạng.")
            raise HTTPException(status_code=400, detail="Invalid image data")
            
        print(f"DEBUG: Decode thành công. Kích thước frame: {frame.shape}")

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

                # Nhận diện (DeepFace)
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
