from django.apps import AppConfig


class DocumentsConfig(AppConfig):
    name = 'server.apps.documents'
    verbose_name = 'Document Processing'
    default_auto_field = 'django.db.models.BigAutoField'

    def ready(self):
        import server.apps.documents.signals  # noqa: F401
