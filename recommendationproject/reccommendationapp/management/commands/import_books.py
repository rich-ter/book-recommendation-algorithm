import csv
import logging
from django.core.management.base import BaseCommand
from django.db import transaction
from reccommendationapp.models import Book

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Import book data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The path to the books CSV file')

    @transaction.atomic
    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        books_to_create = []
        books_to_update = []
        count = 0
        batch_size = 10000  # Larger batch size for efficiency

        # Cache existing books
        existing_books = {book.isbn: book for book in Book.objects.all()}

        with open(csv_file, newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            header = next(reader)  # Skip the header row

            for row in reader:
                try:
                    isbn, title, author, year, publisher, url_s, url_m, url_l = row
                    isbn = isbn.strip()  # Strip spaces from ISBN

                    if isbn in existing_books:
                        # Update existing book
                        book = existing_books[isbn]
                        book.title = title
                        book.author = author
                        book.year_of_publication = int(year)
                        book.publisher = publisher
                        book.image_url_s = url_s
                        book.image_url_m = url_m
                        book.image_url_l = url_l
                        books_to_update.append(book)
                    else:
                        # Create new book
                        books_to_create.append(
                            Book(
                                isbn=isbn,
                                title=title,
                                author=author,
                                year_of_publication=int(year),
                                publisher=publisher,
                                image_url_s=url_s,
                                image_url_m=url_m,
                                image_url_l=url_l
                            )
                        )
                    count += 1

                    # Bulk create or update books in batches
                    if len(books_to_create) >= batch_size:
                        Book.objects.bulk_create(books_to_create)
                        logger.info(f'{len(books_to_create)} book records created...')
                        books_to_create = []

                    if len(books_to_update) >= batch_size:
                        Book.objects.bulk_update(books_to_update, [
                            'title', 'author', 'year_of_publication', 'publisher', 'image_url_s', 'image_url_m', 'image_url_l'
                        ])
                        logger.info(f'{len(books_to_update)} book records updated...')
                        books_to_update = []

                except ValueError as e:
                    logger.error(f'Error processing row {row}: {e}')
                    continue

            # Create or update remaining books
            if books_to_create:
                Book.objects.bulk_create(books_to_create)
                logger.info(f'{len(books_to_create)} book records created...')

            if books_to_update:
                Book.objects.bulk_update(books_to_update, [
                    'title', 'author', 'year_of_publication', 'publisher', 'image_url_s', 'image_url_m', 'image_url_l'
                ])
                logger.info(f'{len(books_to_update)} book records updated...')

        self.stdout.write(self.style.SUCCESS(f'Total {count} book records processed successfully'))
