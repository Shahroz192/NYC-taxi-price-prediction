FROM python:3.10-slim

WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY config.yaml .
COPY src/ /app/src
COPY models/ /app/models

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application with uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5001"]