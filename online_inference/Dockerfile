FROM python:3.9-slim

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt --no-cache-dir && rm requirements*.txt

CMD ["uvicorn", "transport.app:app", "--host", "0.0.0.0", "--port", "8000"]
