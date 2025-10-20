from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


def index(request: HttpRequest) -> HttpResponse:
    """
    Main (or index) view.

    Returns rendered default page to the user.
    Typed with the help of ``django-stubs`` project.
    """
    return render(request, 'main/index.html')


def health(request: HttpRequest) -> HttpResponse:
    """Return a simple HTTP 200 response for container health checks."""
    return HttpResponse('ok', content_type='text/plain')
