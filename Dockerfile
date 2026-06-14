# Sử dụng Python 3.9 slim để tối ưu dung lượng và RAM
FROM python:3.9-slim

# Thiết lập thư mục làm việc cố định
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và xử lý ảnh
# Sử dụng 'libgl1' thay vì 'libgl1-mesa-glx' để tương thích tốt với bản Debian mới
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt PyTorch bản CPU riêng biệt để tránh timeout khi build
# Đây là bước giúp container khởi tạo ổn định nhất với gói 2GB RAM
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements và cài đặt các thư viện phụ thuộc còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn của bạn vào container
COPY . .

# Chạy ứng dụng bằng Uvicorn
# Dùng biến môi trường $PORT do Render tự cấp phát để tránh lỗi "No open ports"
CMD uvicorn app_ai:app --host 0.0.0.0 --port $PORT
