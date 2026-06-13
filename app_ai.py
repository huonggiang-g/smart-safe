from fastapi import FastAPI, Request
import numpy as np
import cv2
import base64
# Import các thư viện AI của bạn ở đây...

app = FastAPI()

# Hàm xử lý nhận diện (lấy từ verify_face.py của bạn)
def process_face(image_np):
    # Dán logic nhận diện của bạn vào đây
    # Trả về: {"recognized": True, "name": "Hương Giang"}
    return {"recognized": True, "name": "Hương Giang"}

@app.post("/recognize")
async def recognize(request: Request):
    data = await request.json()
    # 1. Decode ảnh từ Base64
    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Gọi hàm xử lý
    result = process_face(img_np)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)