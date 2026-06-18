FROM python:3.9-slim

# Cài thư viện hệ thống
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app_ai:app", "--bind", "0.0.0.0:8000", "--timeout", "120", "--preload"]
