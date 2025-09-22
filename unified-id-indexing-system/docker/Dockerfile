FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Copy the scripts
COPY scripts/ ./scripts/

# Copy the configuration files
COPY config/ ./config/

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["python", "src/api/routes.py"]