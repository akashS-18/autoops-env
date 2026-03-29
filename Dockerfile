FROM python:3.11-slim

# HuggingFace Spaces requires port 7860
WORKDIR /app

# Cache-bust: force full rebuild so new code is always picked up
LABEL build_version="2026-03-29-v4"

# Copy all project files
COPY requirements.txt .
COPY pyproject.toml .
COPY models.py .
COPY baseline.py .
COPY inference.py .
COPY client.py .
COPY server/ ./server/
COPY graders/ ./graders/
COPY tasks/ ./tasks/
COPY openenv.yaml .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
