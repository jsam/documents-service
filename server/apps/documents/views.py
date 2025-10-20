from django.shortcuts import render


def upload_ui(request):
    return render(request, 'documents/upload_ui.html')
