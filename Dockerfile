FROM python:3.11-slim

# HuggingFace Spaces requires port 7860
WORKDIR /app

# Copy package files
COPY autoops_env/ autoops_env/

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    pydantic>=2.5.0 \
    httpx>=0.26.0 \
    && pip install --no-cache-dir -e autoops_env/

EXPOSE 7860

CMD ["uvicorn", "autoops_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
