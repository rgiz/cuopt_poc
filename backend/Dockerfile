
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/src
COPY backend /app/backend
# ENV PYTHONPATH=/app
ENV PYTHONPATH="/app/src:/app"
ENV PRIVATE_DATA_DIR=/data/private
EXPOSE 8000
CMD ["python", "backend/main_miles.py"]
