import os
import django
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from django.db.models.functions import Cast
from django.db.models import FloatField, Count, IntegerField
import logging
from django.db import connection, transaction

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendationproject.settings')
django.setup()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from reccommendationapp.models import Book, Rating

book_file_path = r'C:\Users\User\OneDrive\Desktop\masters\chalkidi-warehouses\recommendationproject\data\BX-Books.csv'


def readBookData():
    book_df = pd.read_csv(book_file_path, delimiter=';', encoding='ISO-8859-1', on_bad_lines='skip')
    print("\nData types of books data:")
    print(book_df.info())
    
def checkBooksInDatabase():
    books_count = Book.objects.count()
    print(f'Total number of books in our database: {books_count}')


def check_for_duplicates_books():
    # Κοιτάμε για διπλότυπα βιβλία
    duplicates = Book.objects.values('isbn').annotate(count=Count('isbn')).filter(count__gt=1)
    if duplicates.exists():
        print("Duplicate Books:")
        for duplicate in duplicates:
            print(f"isbn: {duplicate['isbn']}, count: {duplicate['count']}")
    else:
        print("No duplicate books found.")

def remove_duplicate_isbn():
    duplicates = Book.objects.values('isbn').annotate(count=Count('isbn')).filter(count__gt=1)
    duplicate_isbns = [duplicate['isbn'] for duplicate in duplicates]

    for isbn in duplicate_isbns:
        books_with_isbn = Book.objects.filter(isbn=isbn)
        for book in books_with_isbn[1:]:
            book.delete()

    print(f"Removed duplicate books. Kept only the first instance of each ISBN.")

def print_unique_years():
    # Εκτυπώνουμε όλες τις μοναδικές τιμές του γνωρίσματος 'year_of_publication' των βιβλίων
    unique_years = Book.objects.annotate(year_int=Cast('year_of_publication', IntegerField())).values_list('year_int', flat=True).distinct()
    unique_years_list = sorted([year for year in unique_years if year is not None])
    print("Unique Years of Publication:")
    print(unique_years_list)
    return unique_years_list


def create_year_distribution_plot():
    # Create Year Distribution Plot
    year_counts = Book.objects.filter(
        year_of_publication__isnull=False,
        year_of_publication__gte=1900,
        year_of_publication__lte=2005
    ).values_list('year_of_publication', flat=True)

    # Debug: Print the query results
    print(f"Number of books found: {len(year_counts)}")
    if len(year_counts) == 0:
        print("No books found for the specified range.")
        return

    # Convert to pandas Series and count occurrences
    year_counts_series = pd.Series(year_counts).value_counts().sort_index()

    # Plotting
    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 15})  # Set larger plot font size

    # Generate a color map for the bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(year_counts_series)))

    plt.bar(year_counts_series.index, year_counts_series.values, color=colors)
    plt.xlabel('Year of Publication')
    plt.ylabel('Counts')
    plt.title('Year of Publication Distribution of Books (1900-2005)')
    plt.show()


def remove_invalid_books():
    total_books_before = Book.objects.count()
    
    # Delete books with a year of publication before 1900 or after 2005
    Book.objects.filter(year_of_publication__lt=1900).delete()
    Book.objects.filter(year_of_publication__gt=2005).delete()
    
    total_books_after = Book.objects.count()
    
    print(f"Total number of books before removal: {total_books_before}")
    print(f"Removed books published before 1900 and after 2005")
    print(f"Total number of books after removal: {total_books_after}")


def check_duplicate_titles_and_whats_in_db():
    duplicate_books = Book.objects.values('title').annotate(count=Count('title')).filter(count__gt=1).order_by('-count')
    total_duplicates = 0
    total_unique_titles = 0
    
    if duplicate_books.exists():
        print("Duplicate book titles and their number of duplicates:")
        for book in duplicate_books:
            total_duplicates += book['count']  # Count all instances
            total_unique_titles += 1
        excess_titles = total_duplicates - total_unique_titles
        print(f"Total number of unique titles with duplicates: {total_unique_titles}")
        print(f"Total number of duplicate entries: {total_duplicates}")
        print(f"Total number of titles that will get deleted: {excess_titles}")
    else:
        print("No duplicate book titles found.")


def check_null_or_empty_books():
    null_or_empty_isbn_count = Book.objects.filter(isbn__isnull=True).count() + Book.objects.filter(isbn='').count()
    null_or_empty_title_count = Book.objects.filter(title__isnull=True).count() + Book.objects.filter(title='').count()
    null_or_empty_author_count = Book.objects.filter(author__isnull=True).count() + Book.objects.filter(author='').count()
    null_or_empty_publisher_count = Book.objects.filter(publisher__isnull=True).count() + Book.objects.filter(publisher='').count()

    print(f'Total number of books with null or empty ISBN: {null_or_empty_isbn_count}')
    print(f'Total number of books with null or empty title: {null_or_empty_title_count}')
    print(f'Total number of books with null or empty author: {null_or_empty_author_count}')
    print(f'Total number of books with null or empty publisher: {null_or_empty_publisher_count}')

    return {
        'null_or_empty_isbn_count': null_or_empty_isbn_count,
        'null_or_empty_title_count': null_or_empty_title_count,
        'null_or_empty_author_count': null_or_empty_author_count,
        'null_or_empty_publisher_count': null_or_empty_publisher_count,
    }


def print_book_details(title):
    # Query the database for books with the given title
    books = Book.objects.filter(title=title)
    
    if books.exists():
        print(f"Details of books with the title '{title}':")
        for book in books:
            print(f"ISBN: {book.isbn}")
            print(f"Title: {book.title}")
            print(f"Author: {book.author}")
            print(f"Year of Publication: {book.year_of_publication}")
            print(f"Publisher: {book.publisher}")
            print(f"Image URL (S): {book.image_url_s}")
            print(f"Image URL (M): {book.image_url_m}")
            print(f"Image URL (L): {book.image_url_l}")
            print("-" * 40)
    else:
        print(f"No books found with the title '{title}'.")


def find_top_10_duplicate_titles():
    return Book.objects.values('title').annotate(count=Count('isbn')).filter(count__gt=1).order_by('-count')[:10]
    
def print_number_of_ratings_per_title():
    duplicates = find_top_10_duplicate_titles()
    
    if not duplicates:
        print("No duplicate book titles found.")
        return
    
    print("Top 10 duplicate book titles and their total number of ratings:")
    
    for duplicate in duplicates:
        title = duplicate['title']
        books_with_title = Book.objects.filter(title=title)
        total_ratings = Rating.objects.filter(book__isbn__in=books_with_title.values_list('isbn', flat=True)).count()
        
        print(f"Title: {title}, Total Ratings: {total_ratings}")







#### BELOW CONSOLIDATION SCRIPT #####
def find_duplicate_titles():
    return Book.objects.values('title').annotate(count=Count('title')).filter(count__gt=1).order_by('-count')

def select_primary_entry(title):
    return Book.objects.filter(title=title).order_by('-year_of_publication').first()

@transaction.atomic
def update_user_ratings(duplicate_title):
    primary_book = select_primary_entry(duplicate_title)
    books_with_title = Book.objects.filter(title=duplicate_title).exclude(id=primary_book.id)

    for book in books_with_title:
        Rating.objects.filter(book=book).update(book=primary_book)
    logger.info(f"Updated ratings to primary book ISBN: {primary_book.isbn} for title: {duplicate_title}")

@transaction.atomic
def remove_duplicates(duplicate_title):
    primary_book = select_primary_entry(duplicate_title)
    books_to_delete = Book.objects.filter(title=duplicate_title).exclude(id=primary_book.id)
    deleted_count = books_to_delete.count()
    books_to_delete.delete()
    logger.info(f"Removed {deleted_count} duplicate books for title: {duplicate_title}")

def consolidate_all_duplicates():
    duplicates = find_duplicate_titles()
    total_duplicates = 0
    total_titles = len(duplicates)
    
    # Number of books before consolidation
    num_books_before = Book.objects.count()
    logger.info(f"Number of books before consolidation: {num_books_before}")

    for i, duplicate in enumerate(duplicates, start=1):
        title = duplicate['title']
        logger.info(f"Processing {i}/{total_titles}: Consolidating title: {title}")
        update_user_ratings(title)
        remove_duplicates(title)
        total_duplicates += duplicate['count'] - 1  # Count duplicates only, exclude the first instance

    # Number of books after consolidation
    num_books_after = Book.objects.count()
    logger.info(f"Number of books after consolidation: {num_books_after}")

    # Total number of books removed
    total_books_removed = num_books_before - num_books_after
    logger.info(f"Before consolidation: {num_books_before} books")
    logger.info(f"After consolidation: {num_books_after} books")
    logger.info(f"Total number of books removed: {total_books_removed}")

    logger.info("Consolidation complete.")


def get_ratings_count_for_title(title):
    # Find the books with the specified title
    books_with_title = Book.objects.filter(title=title)
    
    if not books_with_title.exists():
        print(f"No books found with the title '{title}'.")
        return

    # Aggregate the ratings for these books
    total_ratings_count = Rating.objects.filter(book__in=books_with_title).count()
    
    print(f"Total number of ratings for the title '{title}': {total_ratings_count}")


if __name__ == "__main__":
    # readBookData()
    checkBooksInDatabase()
    # create_year_distribution_plot()
    # check_null_or_empty_books()
    # duplicate_book_titles()
    # remove_duplicate_isbn()# THAT'S DEFINITELY ONE TO KEEP FOR PRE-PROCESSING.
    # remove_invalid_books() # THAT'S DEFINITELY ONE TO KEEP FOR PRE-PROCESSING. 
    # # check_duplicate_titles_and_whats_in_db()
    # # # print_top_10_duplicate_titles()
    # # print_number_of_ratings_per_title()

    # consolidate_all_duplicates()# THAT'S DEFINITELY ONE TO KEEP FOR PRE-PROCESSING. 

    # titles = [
    #     "Selected Poems",
    #     "Little Women",
    #     "Adventures of Huckleberry Finn",
    #     "Dracula",
    #     "Wuthering Heights",
    #     "The Night Before Christmas",
    #     "The Secret Garden",
    #     "Pride and Prejudice",
    #     "Black Beauty",
    #     "Jane Eyre"
    # ]
    # for title in titles:
    #     get_ratings_count_for_title(title)