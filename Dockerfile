# Dockerfile
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System packages (needed for OpenCV, ffmpeg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install torch (CPU-only by default; if GPU enabled on HF, switch to CUDA wheel automatically)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy repo contents
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
