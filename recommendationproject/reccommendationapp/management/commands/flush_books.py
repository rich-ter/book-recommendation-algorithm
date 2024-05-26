from django.core.management.base import BaseCommand
from reccommendationapp.models import Book

class Command(BaseCommand):
    help = 'Flush the Book table'

    def handle(self, *args, **kwargs):
        Book.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Successfully flushed the Book table'))
