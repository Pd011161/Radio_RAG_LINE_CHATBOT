FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget tar && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ติดตั้ง ngrok v3.x (เปลี่ยนตรงนี้)
RUN wget -O ngrok.tgz https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz \
    && tar -xzf ngrok.tgz \
    && mv ngrok /usr/local/bin \
    && rm ngrok.tgz

COPY . .

EXPOSE 8000

CMD ["/bin/sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & if [ ! -z \"$NGROK_AUTHTOKEN\" ]; then ngrok config add-authtoken $NGROK_AUTHTOKEN; fi && ngrok http 8000 --log=stdout"]
