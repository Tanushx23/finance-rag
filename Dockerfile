FROM python:3.11-slim

WORKDIR /app

# System deps needed to build faiss-cpu / torch wheels cleanly on some images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render (and most hosts) inject $PORT at runtime -- don't hardcode 5000.
ENV PORT=5000
EXPOSE 5000


CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app