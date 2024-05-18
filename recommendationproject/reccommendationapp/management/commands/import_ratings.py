import csv
import logging
from django.core.management.base import BaseCommand
from django.db import transaction
from reccommendationapp.models import Rating, User, Book

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Import rating data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The path to the ratings CSV file')

    @transaction.atomic
    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        ratings_to_create = []
        count = 0
        batch_size = 10000  # Larger batch size for efficiency

        # Cache existing users and books
        existing_users = {user.user_id: user for user in User.objects.all()}
        existing_books = {book.isbn: book for book in Book.objects.all()}

        with open(csv_file, newline='', encoding='latin1') as file:  # Changed encoding to 'latin1'
            reader = csv.reader(file, delimiter=';')
            header = next(reader)  # Skip the header row

            for row in reader:
                try:
                    user_id, isbn, book_rating = row
                    user_id = int(user_id)
                    isbn = isbn.strip()  # Strip spaces from ISBN
                    book_rating = int(book_rating)

                    if user_id in existing_users and isbn in existing_books:
                        user = existing_users[user_id]
                        book = existing_books[isbn]
                        ratings_to_create.append(
                            Rating(
                                user=user,
                                book=book,
                                book_rating=book_rating
                            )
                        )
                        count += 1

                    # Bulk create ratings in batches
                    if len(ratings_to_create) >= batch_size:
                        Rating.objects.bulk_create(ratings_to_create)
                        logger.info(f'{len(ratings_to_create)} rating records created...')
                        ratings_to_create = []

                except ValueError as e:
                    logger.error(f'Error processing row {row}: {e}')
                    continue

            # Create remaining ratings
            if ratings_to_create:
                Rating.objects.bulk_create(ratings_to_create)
                logger.info(f'{len(ratings_to_create)} rating records created...')

        self.stdout.write(self.style.SUCCESS(f'Total {count} rating records processed successfully'))
