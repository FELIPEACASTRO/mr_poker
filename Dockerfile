FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .[dev]

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "apps.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
