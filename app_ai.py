import os
import cv2
import numpy as np
import gdown
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

app = FastAPI()

# 1. TẢI FACENET TỪ DRIVE (Dùng gdown để tránh lỗi file HTML)
FACENET_PATH = "resources/facenet512.h5"
if not os.path.exists(FACENET_PATH):
    print("Đang tải Facenet512 từ Drive...")
    gdown.download(id="1pBCQntSyUsnuRBrKCsFTUMLFq1JrW_wL", output=FACENET_PATH, quiet=False)

# 2. KHỞI TẠO CÁC MÔ HÌNH
# Đảm bảo các file .pt và .pth đã có sẵn trong repo GitHub (dưới 25MB hoặc đã được LFS xử lý)
yolo_pose = YOLO('resources/detection_model/best.pt')
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = yolo_pose(frame, verbose=False)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Anti-Spoofing
            img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
            pred = fas_predict.predict(img_crop_fas, "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")
            
            if pred[0][1] > 0.85:
                return {"status": "real", "score": float(pred[0][1])}
            else:
                return {"status": "spoof", "score": float(pred[0][1])}
                
    return {"status": "no_face"}

@app.get("/")
async def root():
    return {"message": "Server AI đã khởi động và sẵn sàng!"}
