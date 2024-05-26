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

user_file_path = r'C:\Users\User\OneDrive\Desktop\masters\chalkidi-warehouses\recommendationproject\data\BX-Users.csv'

def readUserData():
    users_df = pd.read_csv(user_file_path, delimiter=';', encoding='ISO-8859-1', on_bad_lines='skip')
    print("\nData types of users data:")
    print(users_df.info())

def checkUsersInDatabase():
    user_count = User.objects.count()
    print(f'Total number of users in our database: {user_count}')

def check_for_duplicates():
    duplicates = User.objects.values('user_id').annotate(count=Count('user_id')).filter(count__gt=1)
    if duplicates.exists():
        print("Duplicate Users:")
        for duplicate in duplicates:
            print(f"user_id: {duplicate['user_id']}, count: {duplicate['count']}")
    else:
        print("No duplicate users found.")

def checkNullAge():
    total_users_count = User.objects.count()
    null_age_count = User.objects.filter(age__isnull=True).count()
    null_age_percentage = (null_age_count / total_users_count) * 100 if total_users_count > 0 else 0
    print(f'Total number of users with null ages: {null_age_count}')
    print(f'Percentage of users with null ages: {null_age_percentage:.2f}%')

def print_unique_ages():
    unique_ages = User.objects.annotate(age_float=Cast('age', FloatField())).values_list('age_float', flat=True).distinct()
    unique_ages_list = sorted([age for age in unique_ages if age is not None])
    print("Unique Ages:")
    print(unique_ages_list)
    return unique_ages_list

def update_invalid_ages():
    User.objects.filter(age__lt=10).update(age=None)
    User.objects.filter(age__gt=100).update(age=None)

def checkNullLocation():
    null_location_count = User.objects.filter(location__isnull=True).count()
    print(f'Total number of users with null locations: {null_location_count}')
    return null_location_count

def create_age_distribution_plot():
    age_counts = User.objects.filter(age__isnull=False).values_list('age', flat=True)
    age_counts_series = pd.Series(age_counts).value_counts().sort_index()

    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 15})  
    colors = plt.cm.viridis(np.linspace(0, 1, len(age_counts_series)))

    plt.bar(age_counts_series.index, age_counts_series.values, color=colors)
    plt.xlabel('Age')
    plt.ylabel('Counts')
    plt.title('Age Distribution of Users')
    plt.show()

if __name__ == "__main__":
    # print_user_data_types()
    # check_for_duplicates()
    update_invalid_ages()
    # checkNullLocation()
    # print_unique_ages()
    # create_age_distribution_plot()
