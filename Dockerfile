FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG OPENAI_API_KEY
RUN OPENAI_API_KEY=$(printf '%s' "$OPENAI_API_KEY" | tr -d '\r\n') python build_index.py

# Windows で編集した場合の CRLF を除去して実行権限を付与
RUN sed -i 's/\r//' start.sh && chmod +x start.sh

ENV PORT=8080
EXPOSE $PORT

CMD ["./start.sh"]
