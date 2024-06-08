import os
import django
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from django.db.utils import OperationalError
import logging
import numpy as np

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendationproject.settings')
django.setup()

from reccommendationapp.models import Book, Rating

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ratings data
def load_data():
    try:
        ratings = list(Rating.objects.values('user_id', 'book__isbn', 'book_rating'))
        return pd.DataFrame(ratings)
    except OperationalError as e:
        logger.error(f"Database error: {e}")
        return pd.DataFrame()

# Function to randomly select a user who has rated at least 10 books
def select_random_user_with_enough_ratings(num_ratings=10):
    ratings = load_data()
    user_ratings_count = ratings['user_id'].value_counts()
    users_with_enough_ratings = user_ratings_count[user_ratings_count >= num_ratings].index.tolist()
    if not users_with_enough_ratings:
        return None
    else:
        return np.random.choice(users_with_enough_ratings)

# Content-Based Filtering
def content_based_recommendations(user_id, num_recommendations=10):
    books = pd.DataFrame(list(Book.objects.all().values()))
    ratings = load_data()

    if user_id not in ratings['user_id'].values:
        return books.nlargest(num_recommendations, 'average_rating')

    user_ratings = ratings[ratings['user_id'] == user_id]
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(books['title'])

    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(tfidf_matrix)
    user_books_indices = [books.index[books['isbn'] == isbn].tolist()[0] for isbn in user_ratings['book__isbn']]
    sim_scores = nn.kneighbors(tfidf_matrix[user_books_indices], n_neighbors=len(books), return_distance=False).flatten()
    recommended_indices = [idx for idx in sim_scores if idx not in user_books_indices][:num_recommendations]
    recommended_books = books.iloc[recommended_indices]

    return recommended_books

# Evaluate the content-based filtering model
def evaluate_content_based_filtering(user_id, num_recommendations=10):
    ratings = load_data()
    user_ratings = ratings[ratings['user_id'] == user_id]

    if user_ratings.shape[0] < num_recommendations:
        return None  # Insufficient user ratings available for evaluation

    recommended_books = content_based_recommendations(user_id, num_recommendations)

    if recommended_books.empty:
        return None  # No recommendations available for evaluation

    # Merge user ratings with recommended books on 'isbn' to get actual ratings for recommended books
    merged = pd.merge(recommended_books, user_ratings, left_on='isbn', right_on='book__isbn', how='inner')

    # If there are no user ratings available for evaluation, we can't calculate MSE
    if merged.empty:
        return None

    # Calculate MSE
    mse = mean_squared_error(merged['book_rating'], merged['book_rating'])

    return mse

if __name__ == "__main__":
    num_recommendations = 10

    # Select a random user who has rated at least 10 books
    user_id = select_random_user_with_enough_ratings(num_ratings=num_recommendations)
    if user_id is None:
        print("No user found with at least 10 ratings.")
    else:
        print(f"Selected User ID: {user_id}")

        mse = evaluate_content_based_filtering(user_id, num_recommendations)
        if mse is not None:
            print(f"Mean Squared Error for Content-Based Filtering: {mse:.4f}")
        else:
            print("Insufficient user ratings or no recommendations available for evaluation.")
