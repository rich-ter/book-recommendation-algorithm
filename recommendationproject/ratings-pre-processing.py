import os
import django
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from django.db.models import Count, FloatField
from django.db.models.functions import Cast

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendationproject.settings')
django.setup()

from reccommendationapp.models import Rating

def print_rating_data_types():
    # Εκτύπωση του τύπου δεδομένων για τις βαθμολογίες
    print("Rating Data Types:")
    for field in Rating._meta.get_fields():
        print(f"{field.name}: {field.get_internal_type()}")

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

if __name__ == "__main__":
    # Step 1: Check Data Types
    print_rating_data_types()

    # Step 2: Extract Unique Ratings
    print_unique_ratings()

    # Step 3: Create Rating Distribution Plot
    create_rating_distribution_plot()
