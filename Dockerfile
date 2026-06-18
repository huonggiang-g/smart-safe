FROM python:3.9-slim

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Đảm bảo copy requirements.txt trước để tận dụng Docker Cache
COPY requirements.txt .

# Cài đặt thư viện
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    tensorflow \
    tf-keras \
    deepface \
    ultralytics \
    fastapi \
    uvicorn \
    opencv-python-headless \
    scipy \
    supabase
    
RUN python3 -c "from deepface import DeepFace; DeepFace.build_model('Facenet512')"
# Thiết lập biến môi trường để DeepFace lưu model vào thư mục /app/.deepface
ENV DEEPFACE_HOME=/app
# Tải model Facenet512 về trước để build thành một layer cố định trong image
RUN python3 -c "from deepface import DeepFace; DeepFace.build_model('Facenet512')"
# ------------------------------

# Sau khi cài xong thư viện mới copy toàn bộ code
COPY . .

RUN chmod +x download_models.sh && ./download_models.sh

EXPOSE 8000
ENV OMP_NUM_THREADS=1
# Sửa dòng CMD thành:
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app_ai:app", "--bind", "0.0.0.0:8000", "--timeout", "120", "--worker-tmp-dir", "/dev/shm"]
