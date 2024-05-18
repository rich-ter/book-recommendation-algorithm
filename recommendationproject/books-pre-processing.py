import os
import django
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from django.db.models.functions import Cast
from django.db.models import FloatField, Count, IntegerField

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendationproject.settings')
django.setup()

from reccommendationapp.models import Book

def print_book_data_types():
    # Εκτύπωση του τύπου δεδομένων για τα βιβλία
    print("Book Data Types:")
    for field in Book._meta.get_fields():
        print(f"{field.name}: {field.get_internal_type()}")

def check_for_duplicates_books():
    # Κοιτάμε για διπλότυπα βιβλία
    duplicates = Book.objects.values('isbn').annotate(count=Count('isbn')).filter(count__gt=1)
    if duplicates.exists():
        print("Duplicate Books:")
        for duplicate in duplicates:
            print(f"isbn: {duplicate['isbn']}, count: {duplicate['count']}")
    else:
        print("No duplicate books found.")

def print_unique_years():
    # Εκτυπώνουμε όλες τις μοναδικές τιμές του γνωρίσματος 'year_of_publication' των βιβλίων
    unique_years = Book.objects.annotate(year_int=Cast('year_of_publication', IntegerField())).values_list('year_int', flat=True).distinct()
    unique_years_list = sorted([year for year in unique_years if year is not None])
    print("Unique Years of Publication:")
    print(unique_years_list)
    return unique_years_list


def create_year_distribution_plot():
    # Create Year Distribution Plot
    year_counts = Book.objects.filter(year_of_publication__isnull=False, year_of_publication__gte=1900, year_of_publication__lte=2024).values_list('year_of_publication', flat=True)
    year_counts_series = pd.Series(year_counts).value_counts().sort_index()

    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 15})  # Set larger plot font size

    # Generate a color map for the bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(year_counts_series)))

    plt.bar(year_counts_series.index, year_counts_series.values, color=colors)
    plt.xlabel('Year of Publication')
    plt.ylabel('Counts')
    plt.title('Year of Publication Distribution of Books (1900-2024)')
    plt.show()


def top_10_book_titles():
    # Find the top 10 book titles with the most entries
    top_books = Book.objects.values('title').annotate(count=Count('title')).order_by('-count')[:10]
    print("The 10 book titles with the most entries in the books table are:")
    for book in top_books:
        print(f"Title: {book['title']}, Entries: {book['count']}")


if __name__ == "__main__":
    # Step 1: Check Data Types
    # print_book_data_types()

    # # Step 2: Check for Duplicates
    # check_for_duplicates_books()

    # create_year_distribution_plot()

    top_10_book_titles()

