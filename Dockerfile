# 1. Dùng bản python nhẹ để tiết kiệm dung lượng
FROM python:3.9-slim

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Cài đặt các thư viện hệ thống cần thiết cho OpenCV và AI
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy file requirements trước để tận dụng cache (tăng tốc độ build)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code vào container
COPY . .

# 6. Dòng này quan trọng nhất: Render tự cấp biến $PORT, 
# Docker sẽ dùng nó để chạy app đúng cổng mà Render yêu cầu
CMD uvicorn app_ai:app --host 0.0.0.0 --port $PORT
