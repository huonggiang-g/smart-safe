# Sử dụng phiên bản Python nhẹ để tiết kiệm RAM
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code và model vào container
COPY . .

# Chạy ứng dụng với uvicorn
CMD ["uvicorn", "app_ai:app", "--host", "0.0.0.0", "--port", "10000"]