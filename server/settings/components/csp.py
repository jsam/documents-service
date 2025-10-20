"""
This file contains a definition for Content-Security-Policy headers.

Read more about it:
https://developer.mozilla.org/ru/docs/Web/HTTP/Headers/Content-Security-Policy

We are using `django-csp` to provide these headers.
Docs: https://github.com/mozilla/django-csp
"""

# These values might and will be redefined in `development.py` env:
CSP_SCRIPT_SRC: tuple[str, ...] = ("'self'",)
CSP_IMG_SRC: tuple[str, ...] = ("'self'",)
CSP_FONT_SRC: tuple[str, ...] = ("'self'",)
CSP_STYLE_SRC: tuple[str, ...] = ("'self'",)
CSP_DEFAULT_SRC: tuple[str, ...] = ("'none'",)
# ``django-csp`` leaves ``CSP_CONNECT_SRC`` empty by default which blocks the
# Swagger UI from requesting ``/api/openapi.json``.  Allow connections to the
# current origin so the documentation can load the schema.
CSP_CONNECT_SRC: tuple[str, ...] = ("'self'",)
