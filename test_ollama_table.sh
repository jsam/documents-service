#!/bin/bash
set -e

echo "=== Testing Ollama Table Extraction Setup ==="
echo

echo "1. Checking if Ollama container is running..."
docker compose -f docker-compose.app.yml ps ollama
echo

echo "2. Checking if model is downloaded..."
docker compose -f docker-compose.app.yml exec ollama ollama list
echo

echo "3. Testing Ollama API accessibility from celery container..."
docker compose -f docker-compose.app.yml exec celery curl -s http://ollama:11434/api/tags | head -20
echo

echo "4. Running database migration..."
docker compose -f docker-compose.app.yml run --rm web python manage.py migrate
echo

echo "5. Restarting celery worker..."
docker compose -f docker-compose.app.yml restart celery
echo

echo "=== Setup complete! ==="
echo "Now process a document with tables and check the results."
