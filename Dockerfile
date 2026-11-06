FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/backend/app/ ./app/
COPY deploy/backend/src/ ./src/
COPY deploy/backend/models/ ./models/
COPY deploy/frontend/ /frontend/

EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --ws auto