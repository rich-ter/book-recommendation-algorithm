import os
import django
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
import random

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendationproject.settings')
django.setup()

from reccommendationapp.models import Book, Rating

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ratings data
def load_data():
    logger.info("Loading ratings data...")
    ratings = list(Rating.objects.values('user_id', 'book__isbn', 'book_rating'))
    logger.info("Ratings data loaded successfully.")
    return pd.DataFrame(ratings)

# Build and cache the TF-IDF matrix
def build_tfidf_matrix():
    logger.info("Building TF-IDF matrix...")
    books = pd.DataFrame(list(Book.objects.all().values()))
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(books['title'])
    return tfidf_matrix, books

# Load or compute the Nearest Neighbors model
def load_or_compute_nn(tfidf_matrix):
    logger.info("Computing Nearest Neighbors model for books...")
    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(tfidf_matrix)
    return nn

# Load or compute the SVD model
def load_or_compute_svd(ratings_df):
    logger.info("Computing SVD model for collaborative filtering...")
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_df[['user_id', 'book__isbn', 'book_rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)
    return svd

# Content-Based Filtering
def content_based_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, num_recommendations=10):
    logger.info(f"Generating content-based recommendations for user {user_id}...")
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]

    if user_id not in ratings_df['user_id'].values:
        return books.nlargest(num_recommendations, 'average_rating')

    user_books_indices = [books.index[books['isbn'] == isbn].tolist()[0] for isbn in user_ratings['book__isbn']]
    sim_scores = nn.kneighbors(tfidf_matrix[user_books_indices], n_neighbors=len(books), return_distance=False).flatten()
    recommended_indices = [idx for idx in sim_scores if idx not in user_books_indices][:num_recommendations]
    return books.iloc[recommended_indices]

# Collaborative Filtering with SVD
def collaborative_filtering_recommendations(user_id, ratings_df, svd, num_recommendations=10):
    logger.info(f"Generating collaborative filtering recommendations for user {user_id}...")
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
def hybrid_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, svd, num_recommendations=10):
    logger.info(f"Generating hybrid recommendations for user {user_id}...")
    content_recs = content_based_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, num_recommendations * 2)
    content_rec_isbns = content_recs['isbn'].tolist()

    collaborative_predictions = [svd.predict(user_id, isbn) for isbn in content_rec_isbns]
    collaborative_predictions.sort(key=lambda x: x.est, reverse=True)

    top_books = [prediction.iid for prediction in collaborative_predictions[:num_recommendations]]
    return Book.objects.filter(isbn__in=top_books)

# Function to split dataset into train and test sets based on users with ratings
def split_train_test_set(data, test_size=0.2):
    # Get unique users who have provided ratings
    users_with_ratings = data['user_id'].unique()
    
    # Randomly select a subset of users for the test set
    test_users = random.sample(list(users_with_ratings), int(len(users_with_ratings) * test_size))
    
    # Filter the dataset to include only the selected test users
    test_data = data[data['user_id'].isin(test_users)]
    
    # Exclude the test users from the train set
    train_data = data[~data['user_id'].isin(test_users)]
    
    # Convert train and test sets to Surprise Dataset
    reader = Reader(rating_scale=(1, 10))
    trainset = Dataset.load_from_df(train_data[['user_id', 'book__isbn', 'book_rating']], reader).build_full_trainset()
    testset = Dataset.load_from_df(test_data[['user_id', 'book__isbn', 'book_rating']], reader).build_full_trainset().build_testset()
    
    return trainset, testset

# Evaluate the model
def evaluate_model(testset, svd):
    logger.info("Evaluating model...")
    predictions = [svd.predict(uid, iid).est for uid, iid, _ in testset]
    actuals = [true_r for _, _, true_r in testset]
    mse = mean_squared_error(actuals, predictions)
    logger.info(f"Evaluation completed with MSE: {mse}")
    return mse

if __name__ == "__main__":
    num_recommendations = 5
    user_id = 115045

    # Load data and models
    ratings_df = load_data()
    tfidf_matrix, books = build_tfidf_matrix()
    nn = load_or_compute_nn(tfidf_matrix)
    svd = load_or_compute_svd(ratings_df)
    
    # Content-Based Filtering
    content_recs = content_based_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, num_recommendations)
    print("Content-Based Filtering Recommendations:")
    print(content_recs)

    # Collaborative Filtering
    collaborative_recs = collaborative_filtering_recommendations(user_id, ratings_df, svd, num_recommendations)
    print("\nCollaborative Filtering Recommendations:")
    for rec in collaborative_recs:
        print(rec.title)

    # Hybrid Recommendations
    hybrid_recs = hybrid_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, svd, num_recommendations)
    print("\nHybrid Recommendations:")
    for rec in hybrid_recs:
        print(rec.title)
    # Load data and models
    ratings_df = load_data()
    tfidf_matrix, books = build_tfidf_matrix()
    nn = load_or_compute_nn(tfidf_matrix)
    svd = load_or_compute_svd(ratings_df)
    
    # Split dataset into train and test sets
    trainset, testset = split_train_test_set(ratings_df)
    
    # Evaluate the hybrid model on the test set
    mse_hybrid = evaluate_model(testset, svd)  # Change this line to use hybrid recommendations
    logger.info(f"\nMean Squared Error for Hybrid Recommendations: {mse_hybrid}")