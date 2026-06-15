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

# Cài đặt thư viện (nếu có lỗi, nó sẽ báo ngay tại đây trong build log)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Sau khi cài xong thư viện mới copy toàn bộ code
COPY . .

RUN chmod +x download_models.sh && ./download_models.sh

EXPOSE 8000
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app_ai:app", "--bind", "0.0.0.0:8000", "--timeout", "300"]
