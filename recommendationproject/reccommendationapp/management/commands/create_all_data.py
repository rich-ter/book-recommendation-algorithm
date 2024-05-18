import logging
from django.core.management import call_command
from django.core.management.base import BaseCommand

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Run import commands for users, books, and ratings in sequence'

    def add_arguments(self, parser):
        parser.add_argument('users_csv', type=str, help='The path to the users CSV file')
        parser.add_argument('books_csv', type=str, help='The path to the books CSV file')
        parser.add_argument('ratings_csv', type=str, help='The path to the ratings CSV file')

    def handle(self, *args, **kwargs):
        users_csv = kwargs['users_csv']
        books_csv = kwargs['books_csv']
        ratings_csv = kwargs['ratings_csv']

        try:
            logger.info('Starting import of users...')
            call_command('import_users', users_csv)
            logger.info('Successfully imported users.')

            logger.info('Starting import of books...')
            call_command('import_books', books_csv)
            logger.info('Successfully imported books.')

            logger.info('Starting import of ratings...')
            call_command('import_ratings', ratings_csv)
            logger.info('Successfully imported ratings.')

            self.stdout.write(self.style.SUCCESS('All data imported successfully!'))

        except Exception as e:
            logger.error(f'An error occurred: {e}')
            self.stdout.write(self.style.ERROR('An error occurred while importing data.'))
