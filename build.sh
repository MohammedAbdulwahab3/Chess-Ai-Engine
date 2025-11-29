#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Install Stockfish
apt-get update
apt-get install -y stockfish

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate
