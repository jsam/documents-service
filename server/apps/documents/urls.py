from django.urls import path

from .views import upload_ui

app_name = 'documents'

urlpatterns = [
    path('ui', upload_ui, name='upload_ui'),
]
