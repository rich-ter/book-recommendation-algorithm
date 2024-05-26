import os
import django
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from django.db.models import Count, FloatField
from django.db.models.functions import Cast
import logging

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendationproject.settings')
django.setup()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from reccommendationapp.models import Rating


ratings_file_path = r'C:\Users\User\OneDrive\Desktop\masters\chalkidi-warehouses\recommendationproject\data\BX-Book-Ratings.csv'


def readRatingsData():
    ratings_df = pd.read_csv(ratings_file_path, delimiter=';', encoding='ISO-8859-1', on_bad_lines='skip')
    print("\nData types of books data:")
    print(ratings_df.info())
    
def checkRatingsInDatabase():
    ratings_count = Rating.objects.count()
    print(f'Total number of ratings in our database: {ratings_count}')

def print_unique_ratings():
    # Εκτυπώνουμε όλες τις μοναδικές τιμές του γνωρίσματος 'book_rating' των βαθμολογιών
    unique_ratings = Rating.objects.values_list('book_rating', flat=True).distinct()
    unique_ratings_list = sorted([rating for rating in unique_ratings if rating is not None])
    print("Unique Ratings:")
    print(unique_ratings_list)
    return unique_ratings_list

def create_rating_distribution_plot():
    # Create Rating Distribution Plot
    rating_counts = Rating.objects.values_list('book_rating', flat=True)
    rating_counts_series = pd.Series(rating_counts).value_counts().sort_index()

    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 15})  # Set larger plot font size

    # Generate a color map for the bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(rating_counts_series)))

    plt.bar(rating_counts_series.index, rating_counts_series.values, color=colors)
    plt.xlabel('Book Rating')
    plt.ylabel('Counts')
    plt.title('Book Rating Distribution')
    plt.show()

def remove_zero_ratings():
    zero_ratings_count = Rating.objects.filter(book_rating=0).count()
    logger.info(f"Number of ratings with value 0: {zero_ratings_count}")
    
    Rating.objects.filter(book_rating=0).delete()
    
    logger.info(f"Removed {zero_ratings_count} ratings with value 0 from the database")


if __name__ == "__main__":
    # Step 1: Check Data Types
    # readRatingsData()

    # Step 2: Extract Unique Ratings
    # print_unique_ratings()

    # Step 3: Create Rating Distribution Plot
    # create_rating_distribution_plot()
    # checkRatingsInDatabase()
    # remove_zero_ratings()
    # checkRatingsInDatabase()
    create_rating_distribution_plot()
