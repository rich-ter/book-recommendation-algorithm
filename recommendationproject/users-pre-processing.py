import os
import django
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from django.db.models.functions import Cast
from django.db.models import FloatField, Count
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendationproject.settings')
django.setup()

from reccommendationapp.models import User

def print_user_data_types():
    # Εκτύπωση του ΄τυπου δεδομένων για τους users 
    print("User Data Types:")
    for field in User._meta.get_fields():
        print(f"{field.name}: {field.get_internal_type()}")

def check_for_duplicates():
    # Κοιτάμε για διπλότυπους χρήστες
    duplicates = User.objects.values('user_id').annotate(count=Count('user_id')).filter(count__gt=1)
    if duplicates.exists():
        print("Duplicate Users:")
        for duplicate in duplicates:
            print(f"user_id: {duplicate['user_id']}, count: {duplicate['count']}")
    else:
        print("No duplicate users found.")

def print_unique_ages():
    # Εκτυπώνουμε όλες τις μοναδικές τιμές του γνωρίσματος ηλικίας των χρηστών
    unique_ages = User.objects.annotate(age_float=Cast('age', FloatField())).values_list('age_float', flat=True).distinct()
    unique_ages_list = sorted([age for age in unique_ages if age is not None])
    print("Unique Ages:")
    print(unique_ages_list)
    return unique_ages_list

def update_invalid_ages():
    # Κάνουμε update τις ηλικίες που είναι κάτω απο 10 και πάνω απο 100 να είναι None.
    User.objects.filter(age__lt=10).update(age=None)
    User.objects.filter(age__gt=100).update(age=None)

def create_age_distribution_plot():
    # Create Age Distribution Plot
    age_counts = User.objects.filter(age__isnull=False).values_list('age', flat=True)
    age_counts_series = pd.Series(age_counts).value_counts().sort_index()

    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 15})  # Set larger plot font size

    # Generate a color map for the bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(age_counts_series)))

    plt.bar(age_counts_series.index, age_counts_series.values, color=colors)
    plt.xlabel('Age')
    plt.ylabel('Counts')
    plt.title('Age Distribution of Users')
    plt.show()

if __name__ == "__main__":
    # # Step 1: Check Data Types
    # print_user_data_types()

    # # Step 2: Check for Duplicates
    # check_for_duplicates()

    # # Step 3: Update Ages
    # update_invalid_ages()

    # # Step 4: Extract Unique Ages
    # print_unique_ages()

    # Step 5: Create Age Distribution Plot
    create_age_distribution_plot()
