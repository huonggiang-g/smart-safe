import cv2
import os
import glob
import pickle
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
from collections import deque
from src.anti_spoof_predict import AntiSpoofPredict 
from src.generate_patches import CropImage 

# ==========================================
# 1. CẤU HÌNH & KHỞI TẠO
# ==========================================
THRESHOLD_FACENET = 0.35 
THRESHOLD_FAS = 0.85     
DATABASE_FILE = "face_database.pkl"
VOTING_THRESHOLD = 7     # Cần 7/10 frame là Real mới mở khóa

yolo_pose = YOLO('best.pt') 
fas_predict = AntiSpoofPredict(device_id=-1)
image_cropper = CropImage() 
fas_history = deque(maxlen=10) # Lưu lịch sử 10 frame FAS

# ==========================================
# 2. NẠP DATABASE (Tối ưu với Pickle)
# ==========================================
def load_or_build_database():
    if os.path.exists(DATABASE_FILE):
        print("[INFO] Đang nạp database từ file...")
        with open(DATABASE_FILE, "rb") as f: return pickle.load(f)
    
    print("[INFO] Đang nạp database từ ảnh...")
    db = {}
    for path in glob.glob(r"G:\nam4ki2\thuc_tap\data_face\*.jpg"):
        name = '_'.join(os.path.basename(path).split('_')[:-1])
        try:
            res = DeepFace.represent(img_path=path, model_name="Facenet512", detector_backend="opencv", enforce_detection=False)
            db.setdefault(name, []).append(np.array(res[0]["embedding"]))
        except: continue
    with open(DATABASE_FILE, "wb") as f: pickle.dump(db, f)
    return db

known_faces_db = load_or_build_database()

# ==========================================
# 3. VÒNG LẶP WEBCAM (TỐI ƯU FPS)
# ==========================================
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = yolo_pose(frame, verbose=False)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[max(0, y1):y2, max(0, x1):x2]
            if face_crop.size == 0: continue

            # --- CHỐNG GIẢ MẠO (FAS) VỚI VOTING ---
            img_crop_fas = image_cropper.crop(frame, [x1, y1, x2-x1, y2-y1], 2.7, 80, 80)
            pred = fas_predict.predict(img_crop_fas, "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")
            fas_history.append(1 if pred[0][1] > THRESHOLD_FAS else 0)
            
            if sum(fas_history) < VOTING_THRESHOLD:
                cv2.putText(frame, "SPOOF (Voting)!", (x1, y1-10), 0, 0.6, (0,0,255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                continue 

            # --- NHẬN DIỆN (FaceNet512) ---
            try:
                res = DeepFace.represent(img_path=face_crop, model_name="Facenet512", detector_backend="skip", enforce_detection=False)
                current_vec = np.array(res[0]["embedding"])
                
                best_name, min_dist = "Unknown", 1.0
                for name, vecs in known_faces_db.items():
                    for v in vecs:
                        dist = cosine(current_vec, v)
                        if dist < min_dist:
                            min_dist, best_name = dist, name
                
                if min_dist < THRESHOLD_FACENET:
                    cv2.putText(frame, f"Unlock: {best_name}", (x1, y1-10), 0, 0.6, (0,255,0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x1, y1-10), 0, 0.6, (0,0,255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            except: pass

    cv2.imshow("Ket Sat AIoT - Final Version", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()