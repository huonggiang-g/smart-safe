# Sử dụng Python 3.9 slim để nhẹ image
FROM python:3.9-slim

# Cài đặt các dependencies hệ thống cần thiết cho OpenCV và DeepFace
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy các file từ repo vào container
COPY . .

# Cấp quyền chạy script và tải các mô hình từ Google Drive
RUN chmod +x download_models.sh && ./download_models.sh

# Cài đặt các gói Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Mở port 8000 cho Render
EXPOSE 8000

# Khởi chạy ứng dụng bằng Gunicorn
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app_ai:app", "--bind", "0.0.0.0:8000"]
