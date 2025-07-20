FROM python:3.10-slim-buster

WORKDIR /app

# Copy ml/requirements.txt from host to /app/requirements.txt in container
COPY ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the scripts directory
RUN mkdir -p ml/scripts

# Copy the explore_data.py script from ml/scripts/ to /app/ml/scripts/
COPY ml/scripts/explore_data.py ./ml/scripts/

# Set environment variables for database connection (runtime injection is best practice for production, but ENV for capstone utility is okay)
ENV DB_HOST=host.docker.internal
ENV DB_PORT=5432
ENV DB_NAME=postgres
ENV DB_USER=postgres

# Default command to run the EDA script.
# You can override this CMD when you run the container if you want to run other scripts.
CMD ["python", "ml/scripts/explore_data.py"]