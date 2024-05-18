import csv
import logging
import os
from django.core.management.base import BaseCommand
from django.db import transaction
from reccommendationapp.models import Book

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_csv(input_file, output_file, header):
    with open(input_file, 'r', encoding='latin1') as infile, open(output_file, 'w', encoding='latin1', newline='') as outfile:
        reader = csv.reader(infile, delimiter=';', quotechar='"')
        writer = csv.writer(outfile, delimiter=';', quotechar='"')

        writer.writerow(header)  # Write the header to the output file

        for row in reader:
            # Join the row and then split it back to ensure proper alignment
            joined_row = ';'.join(row)
            split_row = joined_row.split(';')

            # Ensure the row has the correct number of columns by filling missing values with 'N/A'
            if len(split_row) < len(header):
                split_row.extend(['N/A'] * (len(header) - len(split_row)))
            elif len(split_row) > len(header):
                split_row = split_row[:len(header)]
            writer.writerow(split_row)

class Command(BaseCommand):
    help = 'Import book data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The path to the books CSV file')

    @transaction.atomic
    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        temp_csv_file_path = 'temp_books.csv'
        
        # Preprocess CSV to ensure correct alignment
        with open(csv_file, newline='', encoding='latin1') as file:
            reader = csv.reader(file, delimiter=';', quotechar='"')
            header = next(reader)  # Read the header
        preprocess_csv(csv_file, temp_csv_file_path, header)
        logger.info(f'Preprocessed CSV file created: {temp_csv_file_path}')

        books_to_create = []
        books_to_update = []
        count = 0
        batch_size = 10000  # Larger batch size for efficiency

        # Cache existing books
        existing_books = {book.isbn: book for book in Book.objects.all()}

        with open(temp_csv_file_path, newline='', encoding='latin1') as file:
            reader = csv.reader(file, delimiter=';', quotechar='"')
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
                        book.year_of_publication = int(year) if year.isdigit() else None
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
                                year_of_publication=int(year) if year.isdigit() else None,
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
