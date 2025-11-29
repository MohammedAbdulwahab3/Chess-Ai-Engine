FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    stockfish \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Collect static files
RUN python manage.py collectstatic --no-input

# Expose port
EXPOSE 8000

# Start command
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "chess_backend.asgi:application"]
