FROM python:3.12-slim

WORKDIR /app

# Install Python deps
COPY ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy UI code
COPY ui /app/ui

# Helpful defaults
ENV STREAMLIT_SERVER_PORT=8501 \
    PYTHONUNBUFFERED=1

EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

