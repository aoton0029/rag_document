@echo off
echo ========================================
echo Advanced RAG System - Setup Script
echo ========================================
echo.

echo 1. Setting up Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo 2. Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo 3. Creating necessary directories...
if not exist "data" mkdir data
if not exist "data\sample_docs" mkdir data\sample_docs
if not exist "logs" mkdir logs

echo.
echo 4. Copying environment configuration...
if not exist ".env" copy ".env.example" ".env"

echo.
echo 5. Starting Docker services...
docker-compose up -d

echo.
echo 6. Waiting for services to be ready...
timeout /t 30 /nobreak

echo.
echo 7. Checking Ollama installation...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Ollama not found. Please install from https://ollama.com/
    echo After installation, run the following commands:
    echo   ollama serve
    echo   ollama pull llama3.1:8b
    echo   ollama pull nomic-embed-text
) else (
    echo Ollama found. Pulling required models...
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
)

echo.
echo ========================================
echo Setup completed!
echo ========================================
echo.
echo To get started:
echo   1. Ensure Ollama is running: ollama serve
echo   2. Run the demo: python examples\demo.py
echo   3. Or start interactive mode: python main.py --interactive
echo.
echo Check README.md for detailed instructions.
echo.

pause