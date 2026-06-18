import torch
import cv2
import numpy as np
import base64
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
from supabase import create_client
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
import gc

# Patch torch
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
original_torch_load = torch.load
torch.load = patched_torch_load

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# Biến toàn cục
yolo_pose = None
fas_predict = None
image_cropper = None

def load_models():
    global yolo_pose, fas_predict, image_cropper
    if yolo_pose is None:
        print("[INFO] Đang nạp model (Lazy Loading)...")
        yolo_pose = YOLO('models/best.pt')
        yolo_pose.model.float()
        fas_predict = AntiSpoofPredict(device_id=-1)
        image_cropper = CropImage()
        print("[INFO] Nạp model hoàn tất!")

@app.get("/health")
async def health():
    return {"status": "ok"}

class ImageRequest(BaseModel):
    image: str

@app.post("/recognize")
async def recognize(request: ImageRequest):
    load_models()
    try:
        img_data = base64.b64decode(request.image.split(',')[-1] if ',' in request.image else request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        results = yolo_pose(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # FAS
                img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
                pred = fas_predict.predict(img_crop_fas, 'models/2.7_80x80_MiniFASNetV2.pth')
                if float(pred[0][1]) < 0.85:
                    return {"recognized": False, "detected": True, "error": "Spoof detected"}

                # DeepFace - Dùng VGG-Face cho nhẹ RAM
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                res = DeepFace.represent(img_path=face_crop, model_name="VGG-Face", detector_backend="skip", enforce_detection=False)
                current_vec = np.array(res[0]["embedding"])
                
                response = supabase.table("faces").select("name, embedding").execute()
                best_name, min_dist = "Unknown", 0.4 # Ngưỡng VGG-Face thường khắt khe hơn
                for item in response.data:
                    dist = cosine(current_vec, np.array(item["embedding"]))
                    if dist < min_dist:
                        min_dist, best_name = dist, item["name"]
                
                gc.collect()
                return {"recognized": True if best_name != "Unknown" else False, "name": best_name}
        
        return {"recognized": False, "detected": False, "message": "No face detected"}
    except Exception as e:
        return {"error": str(e)}
