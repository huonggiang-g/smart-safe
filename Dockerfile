# Base image nhẹ, ổn định
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy toàn bộ code vào container
COPY . .

# Cài đặt các thư viện (đảm bảo file requirements.txt của bạn đầy đủ)
RUN pip install --no-cache-dir -r requirements.txt

# Cấu hình Gunicorn (Chuẩn cho gói Standard)
# -w 2: 2 workers để xử lý song song
# -k uvicorn.workers.UvicornWorker: Engine của FastAPI
# --preload: Nạp model 1 lần duy nhất vào RAM (cực quan trọng)
# --timeout 120: Tránh lỗi timeout khi nhận diện ảnh nặng
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app_ai:app", "--bind", "0.0.0.0:8000", "--timeout", "120", "--preload"]
