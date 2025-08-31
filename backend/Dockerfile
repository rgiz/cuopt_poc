FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Install your domain deps here or at repo root in one image as you prefer
COPY src /app/src
COPY backend/app.py /app/backend/app.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app 

EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
# FROM python:3.12-slim
# WORKDIR /app
# COPY backend/requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r /app/requirements.txt
# COPY src /app/src
# COPY backend /app/backend
# # ENV PYTHONPATH=/app
# ENV PYTHONPATH="/app/src:/app"
# ENV PRIVATE_DATA_DIR=/data/private
# EXPOSE 8000
# CMD ["python", "backend/main_miles.py"]
