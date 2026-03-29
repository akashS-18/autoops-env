FROM python:3.11-slim

# HuggingFace Spaces requires port 7860
WORKDIR /app

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
