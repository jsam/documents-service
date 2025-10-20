API reference
=============

This project exposes an HTTP API implemented with `django-ninja <https://django-ninja.rest/>`_.
The schema is generated automatically so the documentation always reflects
implemented endpoints. When running the development server you can browse the
interactive API documentation at ``/api/docs`` or download the schema from
``/api/openapi.json``.

All HTTP endpoints require a valid ``X-API-Key`` header, except for the
Swagger UI at ``/api/docs`` and the Django admin interface. The expected
value is configured via the ``API_KEY`` environment variable. In the
interactive documentation click the **Authorize** button and enter the
key so requests issued from the UI are authenticated.

Generating the schema
---------------------

Run the following command before building the documentation to export the
current OpenAPI schema::

    python manage.py export_openapi --file docs/openapi.json

Endpoints
---------

.. openapi:: openapi.json
   :style: table

