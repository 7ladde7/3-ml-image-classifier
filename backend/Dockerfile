FROM python:3.14-slim

WORKDIR /backend

RUN apt-get update && apt-get install -y \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -r requirements.txt

CMD ["fastapi", "dev", "api/routes.py", "--host", "0.0.0.0", "--port", "8000"]