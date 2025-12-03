#!/bin/bash
# Scala 프로젝트 빌드 스크립트

cd "$(dirname "$0")"

echo "Building Scala search engine..."
sbt assembly

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "JAR file location: target/scala-2.13/policy-search-engine.jar"
else
    echo "Build failed!"
    exit 1
fi

