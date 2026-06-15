import cv2
import numpy as np
import base64
import pickle
import os
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

app = FastAPI()

class ImageRequest(BaseModel):
    image: str

# Khởi tạo mô hình
MODEL_YOLO = 'models/best.pt'
MODEL_FAS = 'models/2.7_80x80_MiniFASNetV2.pth'
DB_FILE = "models/face_database.pkl"
THRESHOLD_FACENET = 0.35 

print("[INFO] Đang nạp model & database...")
yolo_pose = YOLO(MODEL_YOLO)
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()

# Load Database
with open(DB_FILE, "rb") as f:
    known_faces_db = pickle.load(f)
print("[INFO] Khởi tạo hoàn tất!")

@app.post("/recognize")
async def recognize(request: ImageRequest):
    try:
        # 1. Decode ảnh
        img_data = base64.b64decode(request.image.split(',')[-1] if ',' in request.image else request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"recognized": False, "detected": False, "error": "Invalid image"}

        # 2. YOLO detect mặt
        results = yolo_pose(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]

                # 3. FAS (Chống giả mạo)
                img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
                pred = fas_predict.predict(img_crop_fas, MODEL_FAS)
                
                if float(pred[0][1]) < 0.85:
                    return {"recognized": False, "detected": True, "error": "Spoof detected"}

                # 4. Nhận diện khuôn mặt (DeepFace)
                res = DeepFace.represent(img_path=face_crop, model_name="Facenet512", detector_backend="skip", enforce_detection=False)
                current_vec = np.array(res[0]["embedding"])
                
                best_name, min_dist = "Unknown", 1.0
                for name, vecs in known_faces_db.items():
                    for v in vecs:
                        dist = cosine(current_vec, v)
                        if dist < min_dist:
                            min_dist, best_name = dist, name
                
                if min_dist < THRESHOLD_FACENET:
                    return {
                        "recognized": True, 
                        "name": best_name, 
                        "confidence": float(1 - min_dist), 
                        "detected": True
                    }
        
        return {"recognized": False, "detected": False}
    except Exception as e:
        return {"recognized": False, "detected": False, "error": str(e)}
