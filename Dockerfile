FROM python:3.10-slim

WORKDIR /app

# Install system-level dependencies required by TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source so this layer is cached
# across source-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY main.py .

# Copy trained artifacts (must exist locally before `docker build`)
# Rebuild the image after retraining to update these
COPY artifacts/battery_life_model.pkl \
     artifacts/scaler.pkl \
     artifacts/label_encoder.pkl \
     ./artifacts/

# Streamlit configuration
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
