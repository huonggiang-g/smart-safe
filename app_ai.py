import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ImageRequest(BaseModel):
    image: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/recognize")
async def recognize(request: ImageRequest):
    # Bước 1: Kiểm tra dữ liệu thô
    print(f"DEBUG: Nhận ảnh, độ dài: {len(request.image)}")
    
    try:
        # Bước 2: Decode cơ bản
        img_data = base64.b64decode(request.image.split(',')[-1] if ',' in request.image else request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Bước 3: Kiểm tra ảnh có rỗng không
        if frame is None:
            return {"status": "error", "message": "cv2 decode failed"}
        
        # Bước 4: Nếu tới đây được, chứng tỏ ảnh hợp lệ
        # Trả về kích thước ảnh để biết server đã xử lý thành công
        return {
            "status": "success", 
            "frame_shape": frame.shape,
            "message": "Ảnh hợp lệ, server AI hoạt động tốt!"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
