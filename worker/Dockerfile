FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV, MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY model_engine.py .
COPY server.py .
COPY worker.py .
COPY __init__.py .

# Models directory (mounted at runtime)
RUN mkdir -p /app/models

EXPOSE 80

ENV MODELS_DIR=/app/models
ENV SERVER_PORT=8000
ENV PYTHONUNBUFFERED=1

# Base image allows commands to be injected via docker-compose.yml
CMD ["bash"]
