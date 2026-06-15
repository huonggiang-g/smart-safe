import cv2
import numpy as np
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

app = FastAPI()

# Cấu trúc JSON nhận từ server.js
class ImageRequest(BaseModel):
    image: str

# Khởi tạo mô hình
print("[INFO] Đang khởi tạo mô hình...")
MODEL_YOLO = 'models/best.pt'
MODEL_FAS = 'models/2.7_80x80_MiniFASNetV2.pth'

yolo_pose = YOLO(MODEL_YOLO)
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage()
print("[INFO] Khởi tạo hoàn tất!")

@app.post("/recognize")
async def recognize(request: ImageRequest):
    try:
        # 1. Decode ảnh từ Base64
        b64_string = request.image
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"recognized": False, "detected": False, "error": "Invalid image"}

        # 2. YOLO nhận diện
        results = yolo_pose(frame, verbose=False)
        
        # 3. Duyệt qua kết quả
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Chống giả mạo (FAS)
                img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
                pred = fas_predict.predict(img_crop_fas, MODEL_FAS)
                score = float(pred[0][1])
                
                # Logic nhận diện (Ví dụ: Nếu real, trả về tên)
                if score > 0.85:
                    # ĐẢM BẢO TRẢ VỀ CẤU TRÚC JSON MÀ SERVER.JS MONG ĐỢI
                    return {
                        "recognized": True, 
                        "name": "Bùi Thị Hương Giang", 
                        "confidence": score, 
                        "detected": True
                    }
                else:
                    return {"recognized": False, "detected": True, "reason": "Spoof detected"}
                    
        return {"recognized": False, "detected": False}

    except Exception as e:
        print(f"[ERROR] AI Processing failed: {str(e)}")
        return {"recognized": False, "detected": False, "error": str(e)}

@app.get("/")
async def root():
    return {"message": "AI Service is Online"}
