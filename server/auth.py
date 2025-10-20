from django.conf import settings
from ninja.security import APIKeyHeader


class APIKeyAuth(APIKeyHeader):
    """Simple API key authentication using ``X-API-Key`` header."""

    param_name = "X-API-Key"

    def authenticate(self, request, key):
        expected = getattr(settings, "API_KEY", None)
        if expected and key == expected:
            return key
        return None
