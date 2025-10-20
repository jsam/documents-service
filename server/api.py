from ninja import NinjaAPI

from server.apps.documents.api.views import router as documents_router

API_TITLE = "Documents Service"
API_VERSION = "1.0.0"

api = NinjaAPI(
    title=API_TITLE,
    version=API_VERSION,
    docs_url="docs",
    openapi_url="openapi.json",
)

api.add_router('/documents', documents_router)
