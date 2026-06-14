import cv2
import numpy as np
import base64
import requests
from fastapi import FastAPI, Request
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

# ================= CẤU HÌNH =================
app = FastAPI()
THRESHOLD_FACENET = 0.35
THRESHOLD_FAS = 0.85
import os
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Load models toàn cục (chỉ chạy 1 lần)
yolo_pose = YOLO('best.pt')
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()

# Load DB từ Supabase
known_faces_db = {}

def load_db_from_supabase():
    global known_faces_db
    try:
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        response = requests.get(f"{SUPABASE_URL}/rest/v1/accounts?select=full_name,face_vector", headers=headers)
        if response.status_code == 200:
            data = response.json()
            known_faces_db = {item['full_name']: [item['face_vector']] for item in data if item['face_vector']}
            print(f"[INFO] Đã nạp {len(known_faces_db)} người từ Supabase")
    except Exception as e:
        print(f"[ERROR] Lỗi nạp DB: {e}")

load_db_from_supabase()

@app.post("/recognize")
async def recognize(request: Request):
    data = await request.json()
    # Decode ảnh
    img_bytes = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = yolo_pose(frame, verbose=False)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Anti-Spoofing (FAS)
            img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
            pred = fas_predict.predict(img_crop_fas, "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")
            if pred[0][1] < THRESHOLD_FAS:
                return {"recognized": False, "message": "SPOOF"}

            # FaceNet512
            face_crop = frame[max(0, y1):y2, max(0, x1):x2]
            res = DeepFace.represent(img_path=face_crop, model_name="Facenet512", detector_backend="skip", enforce_detection=False)
            current_vec = np.array(res[0]["embedding"])

            # So sánh Cosine
            best_name, min_dist = "Unknown", 1.0
            for name, vecs in known_faces_db.items():
                for v in vecs:
                    dist = cosine(current_vec, v)
                    if dist < min_dist:
                        min_dist, best_name = dist, name
            
            if min_dist < THRESHOLD_FACENET:
                return {"recognized": True, "name": best_name, "dist": float(min_dist)}

    return {"recognized": False, "name": "Unknown"}
