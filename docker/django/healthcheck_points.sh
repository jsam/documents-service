#!/bin/sh
# Healthcheck script for POINTS ML worker
# Ensures worker is ready AND model is loaded

# Check if celery worker is responsive
celery -A server inspect ping -d celery@$HOSTNAME -t 5 || exit 1

# Check if model is loaded by checking for marker file
test -f /tmp/points_model_ready || exit 1

exit 0
