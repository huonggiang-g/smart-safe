import cv2
import numpy as np
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

app = FastAPI()

# Định nghĩa cấu trúc dữ liệu JSON từ server.js gửi sang
class ImageRequest(BaseModel):
    image: str

# Khởi tạo mô hình
MODEL_YOLO = 'models/best.pt'
MODEL_FAS = 'models/2.7_80x80_MiniFASNetV2.pth'

print("Đang khởi tạo mô hình...")
yolo_pose = YOLO(MODEL_YOLO)
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()
print("Khởi tạo hoàn tất!")

@app.post("/recognize")
async def recognize(request: ImageRequest):
    try:
        # 1. Decode Base64 string từ JSON
        img_data = base64.b64decode(request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"recognized": False, "error": "Invalid image data"}

        # 2. YOLO phát hiện khuôn mặt
        results = yolo_pose(frame, verbose=False)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 3. Anti-Spoofing
                img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
                pred = fas_predict.predict(img_crop_fas, MODEL_FAS)
                score = float(pred[0][1])
                
                # Trả về kết quả khớp với cấu trúc server.js mong đợi
                # Server.js mong đợi: { recognized, name, confidence, detected }
                if score > 0.85:
                    return {
                        "recognized": True, 
                        "name": "User_Detected", 
                        "confidence": score, 
                        "detected": True
                    }
                else:
                    return {"recognized": False, "detected": True, "error": "Spoof detected"}
                    
        return {"recognized": False, "detected": False}
        
    except Exception as e:
        return {"recognized": False, "error": str(e)}

@app.get("/")
async def root():
    return {"message": "AI Service Ready"}
