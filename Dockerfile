FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
# Sử dụng 2 workers và preload để tối ưu RAM và tốc độ khởi động
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app_ai:app", "--bind", "0.0.0.0:8000", "--timeout", "120", "--preload"]
