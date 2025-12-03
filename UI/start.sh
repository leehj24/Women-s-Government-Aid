#!/bin/bash
# Flask 앱 시작 스크립트 (Linux/Mac)
# Scala 자동 빌드 포함

cd "$(dirname "$0")"

echo "Starting Flask application..."
echo

python3 app.py

if [ $? -ne 0 ]; then
    echo
    echo "Failed to start application!"
    exit 1
fi

