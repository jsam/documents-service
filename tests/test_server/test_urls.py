from http import HTTPStatus

import pytest
from django.test import Client


@pytest.mark.django_db
def test_health_check(client: Client, settings) -> None:
    """Health check is accessible without API key."""
    settings.API_KEY = 'secret'
    response = client.get('/health/')
    assert response.status_code == HTTPStatus.OK
    response = client.get('/health/', HTTP_X_API_KEY='secret')
    assert response.status_code == HTTPStatus.OK


def test_admin_unauthorized(client: Client, settings) -> None:
    """Admin panel redirects to login when no API key is used."""
    settings.API_KEY = 'secret'
    response = client.get('/admin/')

    assert response.status_code == HTTPStatus.FOUND


def test_admin_authorized(admin_client: Client, settings) -> None:
    """Admin panel accessible with API key."""
    settings.API_KEY = 'secret'
    response = admin_client.get('/admin/', HTTP_X_API_KEY='secret')

    assert response.status_code == HTTPStatus.OK


def test_admin_docs_unauthorized(client: Client, settings) -> None:
    """Admin docs redirect to login when no API key is used."""
    settings.API_KEY = 'secret'
    response = client.get('/admin/doc/')

    assert response.status_code == HTTPStatus.FOUND


def test_admin_docs_authorized(admin_client: Client, settings) -> None:
    """Admin docs accessible with API key."""
    settings.API_KEY = 'secret'
    response = admin_client.get('/admin/doc/', HTTP_X_API_KEY='secret')

    assert response.status_code == HTTPStatus.OK
    assert b'docutils' not in response.content


@pytest.mark.parametrize(
    'page',
    [
        '/robots.txt',
        '/humans.txt',
    ],
)
def test_specials_txt(client: Client, settings, page: str) -> None:
    """Special text files require API key."""
    settings.API_KEY = 'secret'
    response = client.get(page)

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    response = client.get(page, HTTP_X_API_KEY='secret')
    assert response.status_code == HTTPStatus.OK
    assert response.get('Content-Type') == 'text/plain'
