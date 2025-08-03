# ml/Dockerfile.ml
# This Dockerfile is for general ML-related Python scripts (EDA, Training, etc.)
# It assumes the scripts will be run from /app/ml/scripts/

FROM python:3.11-slim-bookworm

# Install postgresql-client for psql command in our wait script for data-ingester
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    python3-dev \
    libffi-dev \
    libssl-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements.txt from host to /app/requirements.txt in container
COPY ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the scripts directory
RUN mkdir -p ml/scripts

# Copy the Python scripts from ml/scripts/ to /app/ml/scripts/
COPY ml/scripts/explore_data.py ./ml/scripts/
COPY ml/scripts/train_model.py ./ml/scripts/

# Set environment variables for database connection (runtime injection is best practice for production, but ENV for capstone utility is okay)
ENV DB_HOST=host.docker.internal
ENV DB_PORT=5435
ENV DB_NAME=deploycamp
ENV DB_USER=postgres
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# Default command to run the EDA script.
# This CMD is overridden by the 'ml-training' service in docker-compose.yml
CMD ["python", "ml/scripts/explore_data.py"]