FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for camelot/pdf processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create logs directory
RUN mkdir -p /app/logs

EXPOSE 8080

CMD ["uvicorn", "cifastapi_mosaic:app", "--host", "0.0.0.0", "--port", "8080"]
