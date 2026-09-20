FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir ".[api,cloud]"

CMD ["sh", "-c", "uvicorn nora_quantica.api:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
