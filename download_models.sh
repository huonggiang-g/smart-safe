#!/bin/bash
# Tạo thư mục models nếu chưa có
mkdir -p models

echo "Đang tải model YOLO..."
curl -L -o models/best.pt "https://drive.google.com/uc?export=download&id=1915I0oRP9NtgX3Eh552Wg4WdNNwvf56R"

echo "Đang tải model Anti-Spoofing..."
curl -L -o models/2.7_80x80_MiniFASNetV2.pth "https://drive.google.com/uc?export=download&id=6HWN6bwvnCxxD4BxPkwuyBR5s_aC4Ct_"

echo "Đang tải model FaceNet..."
curl -L -o models/facenet512_weights.h5 "https://drive.google.com/uc?export=download&id=1kAdWmtE7xrGgqe7Nzw3kuSVUFFPZ1DiQ"

echo "Tải xong!"