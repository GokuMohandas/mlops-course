# Base image
FROM python:3.7-slim

# Install dependencies
WORKDIR /mlops
COPY setup.py setup.py
COPY requirements.txt requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -e . --no-cache-dir \
    && python3 -m pip install protobuf==3.20.1 --no-cache-dir \
    && apt-get purge -y --auto-remove gcc build-essential

# Copy
COPY tagifai tagifai
COPY app app
COPY data data
COPY config config
COPY stores stores

# Pull assets from S3
RUN dvc init --no-scm
RUN dvc remote add -d storage stores/blob
RUN dvc pull

# Export ports
EXPOSE 8000

# Start app
ENTRYPOINT ["gunicorn", "-c", "app/gunicorn.py", "-k", "uvicorn.workers.UvicornWorker", "app.api:app"]
