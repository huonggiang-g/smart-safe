@app.post("/recognize")
async def recognize(request: ImageRequest):
    load_models() # Gọi hàm nạp model
    
    try:
        img_data = base64.b64decode(request.image.split(',')[-1] if ',' in request.image else request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"status": "error", "message": "cv2 decode failed"}

        # 1. Phát hiện khuôn mặt bằng YOLO
        results = yolo_pose(frame, verbose=False)
        detected = False
        
        for result in results:
            if len(result.boxes) > 0:
                detected = True
                # Lấy tọa độ khuôn mặt đầu tiên tìm thấy
                box = result.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Trả về kết quả: đã tìm thấy khuôn mặt, kèm tọa độ để web hiển thị (nếu cần)
                return {
                    "status": "success",
                    "detected": True,
                    "box": [x1, y1, x2, y2],
                    "message": "YOLO đã tìm thấy khuôn mặt!"
                }
        
        return {"status": "success", "detected": False, "message": "Không tìm thấy khuôn mặt"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
