# Use PyTorch + CUDA 12.1 runtime image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all your code
COPY . .

# Environment variable for GCS credentials
# If you mount credentials, uncomment the next line
# ENV GOOGLE_APPLICATION_CREDENTIALS=/app/creds.json

# Default command: run your fine-tuning script
CMD ["python", "lora_dict_collect_gcp.py"]