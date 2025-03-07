#!/bin/bash

for port in 12345 12346 12347 12348 12349
do
    echo "Starting FastAPI server on port $port..."
    python ocr_server.py --port $port &
done
python pdf_server.py &
# 等待所有后台进程完成
wait
