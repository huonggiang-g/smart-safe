import os
import requests
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

app = FastAPI()

# 1. TỰ ĐỘNG TẢI MÔ HÌNH (GỌI MỘT LẦN KHI KHỞI ĐỘNG)
def download_model(file_id, dest):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        with open(dest, "wb") as f:
            f.write(requests.get(url).content)

download_model("1IK7hqsybAQ0i7k8cA0HRrtUYk3eZe5O8", "/app/resources/detection_model/best.pt")
download_model("1pBCQntSyUsnuRBrKCsFTUMLFq1JrW_wL", "/app/resources/facenet512.h5")
# BẠN ĐIỀN ID CỦA FILE MINI-FAS-NET VÀO ĐÂY
download_model("ID_FILE_MINI_FAS_NET", "/app/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")

# 2. KHỞI TẠO AI (GLOBAL)
yolo_pose = YOLO('/app/resources/detection_model/best.pt')
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()

@app.post("/predict")
async def predict_face(file: UploadFile = File(...)):
    # Đọc ảnh từ ESP32 gửi lên
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Logic nhận diện (Trích xuất từ main.py của bạn)
    results = yolo_pose(frame, verbose=False)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Kiểm tra FAS
            img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
            pred = fas_predict.predict(img_crop_fas, "/app/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")
            
            # Kết quả (ví dụ: pred[0][1] > 0.85 là Real)
            if pred[0][1] > 0.85:
                return {"status": "real", "message": "Khuôn mặt hợp lệ"}
            else:
                return {"status": "spoof", "message": "Phát hiện giả mạo!"}
                
    return {"status": "no_face", "message": "Không tìm thấy khuôn mặt"}
