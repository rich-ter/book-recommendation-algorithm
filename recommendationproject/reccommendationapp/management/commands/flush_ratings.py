from django.core.management.base import BaseCommand
from reccommendationapp.models import Rating

class Command(BaseCommand):
    help = 'Flush the Book table'

    def handle(self, *args, **kwargs):
        Rating.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Successfully flushed the Rating table'))
