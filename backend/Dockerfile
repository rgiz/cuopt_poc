FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1

# Backend deps
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# App code
COPY src /app/src
COPY backend /app/backend

# (Optional) UI deps in backend image only if you truly need them here — usually you don't.
# If you keep them:
# COPY ui/requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000
CMD ["uvicorn", "backend.main_miles:app", "--host", "0.0.0.0", "--port", "8000"]
