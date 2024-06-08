import os
import django
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from django.core.cache import cache
from django.db.utils import OperationalError
import logging
import numpy as np
import random
import multiprocessing

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendationproject.settings')
django.setup()

from reccommendationapp.models import Book, Rating, User

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

# Calculate average rating for books
def calculate_average_ratings(books, ratings):
    average_ratings = ratings.groupby('book__isbn')['book_rating'].mean().reset_index()
    average_ratings.columns = ['isbn', 'average_rating']
    return books.merge(average_ratings, on='isbn', how='left')

# Content-Based Filtering
def content_based_recommendations(user_id, num_recommendations=10):
    books = pd.DataFrame(list(Book.objects.all().values()))
    ratings = load_data()

    # Ensure average_rating exists
    books = calculate_average_ratings(books, ratings)

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
    return books.iloc[recommended_indices]

# Collaborative Filtering with SVD
def collaborative_filtering_recommendations(user_id, num_recommendations=10):
    svd = load_or_compute_svd()
    ratings_df = load_data()
    
    if ratings_df.empty:
        return []

    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    if user_ratings.empty:
        books = pd.DataFrame(list(Book.objects.all().values()))
        return books.nlargest(num_recommendations, 'average_rating')

    all_books = ratings_df['book__isbn'].unique()
    unrated_books = [isbn for isbn in all_books if isbn not in user_ratings['book__isbn'].values]

    predictions = [svd.predict(user_id, isbn) for isbn in unrated_books]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_books = [prediction.iid for prediction in predictions[:num_recommendations]]
    return Book.objects.filter(isbn__in=top_books)

# Hybrid Recommendation
def hybrid_recommendations(user_id, num_recommendations=10):
    nn = load_or_compute_nn()
    svd = load_or_compute_svd()

    content_recs = content_based_recommendations(user_id, num_recommendations * 2)
    content_rec_isbns = content_recs['isbn'].tolist()

    collaborative_predictions = [svd.predict(user_id, isbn).est for isbn in content_rec_isbns]

    hybrid_scores = [a * 0.5 + b * 0.5 for a, b in zip(content_recs['average_rating'], collaborative_predictions)]
    hybrid_recommendations = content_recs.iloc[np.argsort(hybrid_scores)[-num_recommendations:]]
    
    return hybrid_recommendations

# Model caching functions
def load_or_compute_nn():
    nn = cache.get('nn_model')
    book_indices = cache.get('book_indices')
    if nn is None or book_indices is None:
        books = pd.DataFrame(list(Book.objects.all().values()))
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(books['title'])

        nn = NearestNeighbors(metric='cosine', algorithm='brute')
        nn.fit(tfidf_matrix)

        book_indices = {isbn: idx for idx, isbn in enumerate(books['isbn'])}
        cache.set('nn_model', nn)
        cache.set('book_indices', book_indices)
    return nn

def load_or_compute_svd():
    svd = cache.get('svd_model')
    if svd is None:
        ratings_df = load_data()
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(ratings_df[['user_id', 'book__isbn', 'book_rating']], reader)
        trainset = data.build_full_trainset()
        svd = SVD()
        svd.fit(trainset)
        cache.set('svd_model', svd)
    return svd

# Evaluate the model
def evaluate_model(testset):
    svd = load_or_compute_svd()
    predictions = [svd.predict(uid, iid).est for uid, iid, _ in testset]
    actuals = [true_r for _, _, true_r in testset]
    return mean_squared_error(actuals, predictions)

# Evaluate the collaborative filtering model
def evaluate_collaborative_filtering_model():
    ratings_df = load_data()
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_df[['user_id', 'book__isbn', 'book_rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    mse = evaluate_model(testset)
    return mse

# Calculate MSE for a subset of users using the hybrid method
def calculate_hybrid_mse(num_users=5, num_recommendations=10):
    svd = load_or_compute_svd()  # Ensure the SVD model is loaded or computed
    ratings_df = load_data()
    test_users = random.sample(list(ratings_df['user_id'].unique()), min(num_users, len(ratings_df['user_id'].unique())))
    testset = []

    for user_id in test_users:
        content_recs = content_based_recommendations(user_id, num_recommendations * 2)
        content_rec_isbns = content_recs['isbn'].tolist()

        collaborative_predictions = [svd.predict(user_id, isbn) for isbn in content_rec_isbns]
        collaborative_predictions.sort(key=lambda x: x.est, reverse=True)

        user_actual_ratings = ratings_df[ratings_df['user_id'] == user_id]

        for pred in collaborative_predictions[:num_recommendations]:
            true_rating = user_actual_ratings[user_actual_ratings['book__isbn'] == pred.iid]['book_rating']
            if not true_rating.empty:
                testset.append((user_id, pred.iid, true_rating.values[0]))

    if not testset:
        return float('inf')  # Return infinity to indicate an issue

    return evaluate_model(testset)

if __name__ == "__main__":
    num_recommendations = 10

    # Get hybrid recommendations for a specific user
    user_id = 115045  # Example user_id
    hybrid_recs = hybrid_recommendations(user_id, num_recommendations)
    print("\nHybrid Recommendations:")
    for book in hybrid_recs.itertuples():
        print(f"Title: {book.title}, Author: {book.author}, Year: {book.year_of_publication}, Publisher: {book.publisher}")

    # Evaluate the collaborative filtering model
    mse_collaborative = evaluate_collaborative_filtering_model()
    print(f"\nMean Squared Error for Collaborative Filtering: {mse_collaborative}")

    # Evaluate the hybrid model
    mse_hybrid = calculate_hybrid_mse()
    print(f"\nMean Squared Error for Hybrid Model: {mse_hybrid}")
