# Sử dụng phiên bản Python nhẹ
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cập nhật để cài đặt thư viện thay thế (libgl1 thay vì libgl1-mesa-glx)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code và model
COPY . .

# Chạy ứng dụng
CMD ["uvicorn", "app_ai:app", "--host", "0.0.0.0", "--port", "10000"]
