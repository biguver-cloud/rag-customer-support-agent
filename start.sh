#!/bin/bash
set -e

# FastAPI バックエンドをバックグラウンドで起動（内部 port 8000）
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Streamlit フロントエンドをフォアグラウンドで起動（Cloud Run の $PORT を使用）
exec streamlit run app.py \
    --server.port="${PORT:-8080}" \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableWebsocketCompression=false
