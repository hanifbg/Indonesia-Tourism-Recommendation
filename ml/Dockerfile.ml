FROM python:3.11-slim-bookworm

# Install build-essential and other necessary libraries
# build-essential includes gcc, g++, make, etc.
# libpq-dev is for psycopg2-binary
# python3-dev is for Python headers
# libffi-dev (often needed by some packages)
# libssl-dev (often needed by some packages like cryptography or database drivers)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ml/requirements.txt from host to /app/requirements.txt in container
COPY ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the scripts directory
RUN mkdir -p ml/scripts

# Copy the explore_data.py script from ml/scripts/ to /app/ml/scripts/
COPY ml/scripts/explore_data.py ./ml/scripts/
COPY ml/scripts/train_model.py ./ml/scripts/

# Set environment variables for database connection (runtime injection is best practice for production, but ENV for capstone utility is okay)
ENV DB_HOST=host.docker.internal
ENV DB_PORT=5435
ENV DB_NAME=deploycamp
ENV DB_USER=postgres

# Default command to run the EDA script.
# You can override this CMD when you run the container if you want to run other scripts.
CMD ["python", "ml/scripts/explore_data.py"]