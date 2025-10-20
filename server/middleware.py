from django.conf import settings
from django.http import HttpResponse


class APIKeyMiddleware:
    """Middleware enforcing API key on all requests."""

    EXEMPT_PATH_PREFIXES = (
        '/api/docs',
        '/api/openapi.json',
        '/admin',
        '/health/',
    )

    def __init__(self, get_response):
        """Initialize middleware with the next handler."""
        self.get_response = get_response

    def __call__(self, request):
        """Check API key header and authorize requests."""
        expected = getattr(settings, 'API_KEY', None)
        if expected and not self._is_exempt(request.path):
            key = request.headers.get('X-API-Key')
            if key != expected:
                return HttpResponse('Unauthorized', status=401)
        return self.get_response(request)

    def _is_exempt(self, path: str) -> bool:
        return any(
            path.startswith(prefix)
            for prefix in self.EXEMPT_PATH_PREFIXES
        )
