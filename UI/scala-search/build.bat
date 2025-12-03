@echo off
REM Scala 프로젝트 빌드 스크립트 (Windows)

cd /d "%~dp0"

echo Building Scala search engine...
call sbt assembly

if %ERRORLEVEL% EQU 0 (
    echo Build successful!
    echo JAR file location: target\scala-2.13\policy-search-engine.jar
) else (
    echo Build failed!
    exit /b 1
)

