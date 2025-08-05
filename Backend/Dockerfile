FROM python:3.12.1 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install --upgrade pip
RUN .venv/bin/pip install -r requirements.txt

# --- Final image
FROM python:3.12.1-slim

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY . .

CMD ["python", "app.py"]
