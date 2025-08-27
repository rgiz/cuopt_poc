
FROM python:3.12-slim
WORKDIR /app
COPY ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY ui /app/ui
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8501
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
