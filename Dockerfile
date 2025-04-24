# Use a Python 3.12 slim image
FROM python:3.12

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 

# Set the working directory
WORKDIR /app

# Install system dependencies and Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        g++ \
        python3-dev \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        libjpeg-dev \
        libpng-dev \
        tesseract-ocr \
        tesseract-ocr-eng \
        ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
#     && pip install --no-cache-dir cython==3.0.0 

# Copy requirements file
COPY requirements.txt /app/

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Expose the FastAPI app's default port
EXPOSE 8000

# Command to run the FastAPI app using Gunicorn with Uvicorn worker
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
