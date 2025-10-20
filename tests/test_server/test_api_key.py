import pytest
from django.test import Client


@pytest.mark.django_db
def test_api_docs_no_key(client: Client, settings):
    settings.API_KEY = "secret"
    response = client.get("/api/openapi.json")
    assert response.status_code == 200


@pytest.mark.django_db
def test_api_with_valid_key(client: Client, settings):
    settings.API_KEY = "secret"
    response = client.get("/api/openapi.json", HTTP_X_API_KEY="secret")
    assert response.status_code == 200


@pytest.mark.django_db
def test_openapi_has_auth_scheme(client: Client, settings):
    settings.API_KEY = "secret"
    response = client.get("/api/openapi.json", HTTP_X_API_KEY="secret")
    assert response.status_code == 200
    spec = response.json()
    assert "APIKeyAuth" in spec["components"]["securitySchemes"]
