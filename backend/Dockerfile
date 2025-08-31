FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
# Make /app and /app/src importable
ENV PYTHONPATH=/app:/app/src

# ---- Install backend deps
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# ---- Copy source code (ENTIRE backend, and src)
COPY backend /app/backend
COPY src /app/src

EXPOSE 8000
CMD ["uvicorn", "backend.main_miles:app", "--host", "0.0.0.0", "--port", "8000"]
