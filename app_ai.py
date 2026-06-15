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

# Import các module từ dự án của bạn
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

# Khởi tạo FastAPI
app = FastAPI()

# 1. CẤU HÌNH BIẾN MÔI TRƯỜNG & SUPABASE
# Hãy đảm bảo bạn đã thêm SUPABASE_URL và SUPABASE_KEY vào mục Environment trên Render
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# 2. KHỞI TẠO MÔ HÌNH (Singleton - Nạp 1 lần duy nhất)
print("[INFO] Đang nạp model và kết nối Database...")
MODEL_YOLO = 'models/best.pt'
MODEL_FAS = 'models/2.7_80x80_MiniFASNetV2.pth'
THRESHOLD_FACENET = 0.35 

yolo_pose = YOLO(MODEL_YOLO)
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()
print("[INFO] Khởi tạo hoàn tất, sẵn sàng nhận request!")

class ImageRequest(BaseModel):
    image: str

@app.post("/recognize")
async def recognize(request: ImageRequest):
    try:
        # A. Xử lý ảnh đầu vào
        img_data = base64.b64decode(request.image.split(',')[-1] if ',' in request.image else request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # B. YOLO detect mặt
        results = yolo_pose(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # C. Chống giả mạo (FAS)
                img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
                pred = fas_predict.predict(img_crop_fas, MODEL_FAS)
                
                if float(pred[0][1]) < 0.85:
                    return {"recognized": False, "detected": True, "error": "Spoof detected"}

                # D. DeepFace lấy vector (skip detector vì đã có YOLO)
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                res = DeepFace.represent(img_path=face_crop, model_name="Facenet512", detector_backend="skip", enforce_detection=False)
                current_vec = np.array(res[0]["embedding"])
                
                # E. Truy vấn Supabase thay vì đọc file .pkl
                response = supabase.table("faces").select("name, embedding").execute()
                db_data = response.data 

                best_name, min_dist = "Unknown", 1.0
                for item in db_data:
                    # Chuyển đổi vector từ database (list) thành numpy array
                    v = np.array(item["embedding"])
                    dist = cosine(current_vec, v)
                    if dist < min_dist:
                        min_dist, best_name = dist, item["name"]
                
                # F. Trả về kết quả
                if min_dist < THRESHOLD_FACENET:
                    return {
                        "recognized": True, 
                        "name": best_name, 
                        "confidence": float(1 - min_dist), 
                        "detected": True
                    }
        
        return {"recognized": False, "detected": False}

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {"recognized": False, "detected": False, "error": str(e)}
