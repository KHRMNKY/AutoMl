FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip 

WORKDIR /app

COPY . .
COPY requirements.txt .

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

EXPOSE 8081

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8081"]
