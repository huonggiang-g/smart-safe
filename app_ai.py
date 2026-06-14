import os
import requests
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

app = FastAPI()

# Hàm tải mô hình từ Drive
def download_model(file_id, dest):
    if not os.path.exists(dest):
        print(f"Đang nạp model: {dest}...")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # Thêm tham số confirm=t để bỏ qua cảnh báo quét virus của Google Drive
        url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
        response = requests.get(url)
        with open(dest, "wb") as f:
            f.write(response.content)
        print("Nạp thành công!")
        file_size = os.path.getsize("/app/resources/detection_model/best.pt")
        print(f"Kích thước file best.pt hiện tại là: {file_size} bytes")
# Tải bộ 3 mô hình
download_model("1IK7hqsybAQ0i7k8cA0HRrtUYk3eZe5O8", "/app/resources/detection_model/best.pt")
download_model("1pBCQntSyUsnuRBrKCsFTUMLFq1JrW_wL", "/app/resources/facenet512.h5")
download_model("14jxLLr8YWzwZjyNC_ox6zPEVj4kPgoTu", "/app/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")

# Khởi tạo các class
yolo_pose = YOLO('/app/resources/detection_model/best.pt')
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()

@app.post("/predict")
async def predict_face(file: UploadFile = File(...)):
    # Đọc ảnh gửi từ ESP32/Client
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. Phát hiện khuôn mặt bằng YOLO
    results = yolo_pose(frame, verbose=False)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 2. Xử lý Anti-Spoofing (FAS)
            img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
            pred = fas_predict.predict(img_crop_fas, "/app/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")
            
            # 3. Trả về kết quả
            # pred[0][1] là score của lớp Real
            if pred[0][1] > 0.85:
                return {"status": "real", "score": float(pred[0][1])}
            else:
                return {"status": "spoof", "score": float(pred[0][1])}
                
    return {"status": "no_face"}
