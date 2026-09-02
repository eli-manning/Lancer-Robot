FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY worker.py .
# Unbuffered, or the logs of a worker that sleeps for half an hour at a time
# arrive in a lump long after anyone stopped watching.
ENV PYTHONUNBUFFERED=1
CMD ["python", "worker.py"]
