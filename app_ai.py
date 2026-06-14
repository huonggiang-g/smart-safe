import cv2
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

app = FastAPI()

# --- KHỞI TẠO MÔ HÌNH (Đường dẫn cố định tới thư mục models đã tải về) ---
# Đảm bảo các file này đã được tải vào folder 'models' bởi Dockerfile
MODEL_YOLO = 'models/best.pt'
MODEL_FAS = 'models/2.7_80x80_MiniFASNetV2.pth'

print("Đang khởi tạo mô hình...")
yolo_pose = YOLO(MODEL_YOLO)
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()
print("Khởi tạo hoàn tất!")

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    # Đọc ảnh từ request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"status": "error", "message": "Invalid image"}

    # YOLO phát hiện khuôn mặt
    results = yolo_pose(frame, verbose=False)
    
    # Chỉ xử lý khuôn mặt đầu tiên tìm thấy
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Anti-Spoofing
            img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
            pred = fas_predict.predict(img_crop_fas, MODEL_FAS)
            
            # Trả về kết quả
            score = float(pred[0][1])
            if score > 0.85:
                return {"status": "real", "score": score}
            else:
                return {"status": "spoof", "score": score}
                
    return {"status": "no_face"}

@app.get("/")
async def root():
    return {"message": "Smart-Safe AI Server is running!"}
