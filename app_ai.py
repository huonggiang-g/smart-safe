from fastapi import FastAPI, Request
from ultralytics import YOLO
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
import cv2
import numpy as np
import base64

app = FastAPI()

# Nạp model lúc khởi động (đường dẫn chuẩn như bạn đã có)
yolo_pose = YOLO('resources/detection_model/best.pt')
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()

@app.post("/process-face")
async def process_face(request: Request):
    try:
        data = await request.json()
        img_data = base64.b64decode(data["image"])
        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        # 1. YOLO nhận diện
        results = yolo_pose(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 2. FAS chống giả mạo
                img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
                pred = fas_predict.predict(img_crop_fas, 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth')
                
                if float(pred[0][1]) < 0.85:
                    return {"status": "spoof"} # Trả về lỗi nếu là ảnh giả
                
                # 3. Cắt mặt và encode base64
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                _, buffer = cv2.imencode('.jpg', face_crop)
                b64_face = base64.b64encode(buffer).decode('utf-8')
                
                return {"status": "ok", "face": b64_face}
        
        return {"status": "no_face"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health():
    return {"status": "ok"}
