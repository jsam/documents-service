from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Create a superuser if one does not exist'

    def handle(self, *args, **options):
        User = get_user_model()
        
        if User.objects.filter(is_superuser=True).exists():
            self.stdout.write('Superuser already exists')
            return
        
        username = 'admin'
        email = 'admin@documents-service.local'
        password = 'admin'
        
        User.objects.create_superuser(
            username=username,
            email=email,
            password=password,
        )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Superuser created: {username} / {password}'
            )
        )
