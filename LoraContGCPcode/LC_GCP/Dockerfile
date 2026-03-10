# Base image with miniconda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment file first for caching
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Activate environment automatically
SHELL ["conda", "run", "-n", "progen2_env", "/bin/bash", "-c"]

# Copy project files
COPY . .

# Run script
CMD ["python", "lora_dict_collect_gcp.py"]