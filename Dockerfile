FROM python:3.9-slim

# Cài thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install tf-keras

# Dùng 1 Worker để cực kỳ an toàn về RAM cho gói Standard
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app_ai:app", "--bind", "0.0.0.0:8000", "--timeout", "120", "--preload"]
