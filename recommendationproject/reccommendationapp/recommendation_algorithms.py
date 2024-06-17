import os
import sys
import django
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD
from sklearn.metrics import mean_squared_error
import logging
import random
from django.db.models import Avg
import numpy as np

# Set up Django environment
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

print("Project Path:", project_path)
print("Python Path:", sys.path)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendationproject.settings')
django.setup()

from reccommendationapp.models import User, Rating, Book

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    logger.info("Loading ratings data...")
    ratings = list(Rating.objects.values('user_id', 'book__isbn', 'book_rating'))
    logger.info("Ratings data loaded successfully.")
    return pd.DataFrame(ratings)

def compute_average_ratings():
    logger.info("Computing average ratings for each book...")
    avg_ratings = Rating.objects.values('book__isbn').annotate(avg_rating=Avg('book_rating'))
    books = pd.DataFrame(list(Book.objects.all().values()))
    avg_ratings_df = pd.DataFrame(list(avg_ratings))
    books = books.merge(avg_ratings_df, left_on='isbn', right_on='book__isbn', how='left')
    books['avg_rating'].fillna(0, inplace=True)
    return books

def build_tfidf_matrix():
    logger.info("Building TF-IDF matrix...")
    books = compute_average_ratings()
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(books['title'])
    return tfidf_matrix, books

def load_or_compute_nn(tfidf_matrix):
    logger.info("Computing Nearest Neighbors model for books...")
    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(tfidf_matrix)
    return nn

def create_user_profile(user_id, ratings_df, tfidf_matrix, books):
    logger.info(f"Creating user profile for user {user_id}...")
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    positive_ratings = user_ratings[user_ratings['book_rating'] >= 6]
    
    positive_book_indices = [books.index[books['isbn'] == isbn].tolist()[0] for isbn in positive_ratings['book__isbn']]
    
    if not positive_book_indices:
        return None
    user_profile = tfidf_matrix[positive_book_indices].mean(axis=0)
    
    logger.info(f"User profile created for user {user_id}")
    return np.asarray(user_profile)  

def content_based_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, num_recommendations=10):
    logger.info(f"Generating content-based recommendations for user {user_id}...")
    user_profile = create_user_profile(user_id, ratings_df, tfidf_matrix, books)
    
    if user_profile is None:
        return books.nlargest(num_recommendations, 'avg_rating')

    sim_scores = nn.kneighbors(user_profile, n_neighbors=len(books), return_distance=False).flatten()
    
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    user_books_indices = [books.index[books['isbn'] == isbn].tolist()[0] for isbn in user_ratings['book__isbn']]
    
    recommended_indices = [idx for idx in sim_scores if idx not in user_books_indices][:num_recommendations]
    
    return books.iloc[recommended_indices]

def load_or_compute_svd(ratings_df):
    logger.info("Computing SVD model for collaborative filtering...")
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_df[['user_id', 'book__isbn', 'book_rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD(random_state=SEED)
    svd.fit(trainset)
    return svd

def collaborative_filtering_recommendations(user_id, ratings_df, svd, num_recommendations=10):
    logger.info(f"Generating collaborative filtering recommendations for user {user_id}...")
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    if user_ratings.empty:
        books = compute_average_ratings()
        return books.nlargest(num_recommendations, 'avg_rating')

    all_books = ratings_df['book__isbn'].unique()
    unrated_books = [isbn for isbn in all_books if isbn not in user_ratings['book__isbn'].values]

    predictions = [svd.predict(user_id, isbn) for isbn in unrated_books]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_books = [prediction.iid for prediction in predictions[:num_recommendations]]
    return Book.objects.filter(isbn__in=top_books)

def hybrid_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, svd, num_recommendations=10):
    logger.info(f"Generating hybrid recommendations for user {user_id}...")
    content_recs = content_based_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, num_recommendations * 2)
    content_rec_isbns = content_recs['isbn'].tolist()

    collaborative_predictions = [svd.predict(user_id, isbn) for isbn in content_rec_isbns]
    collaborative_predictions.sort(key=lambda x: x.est, reverse=True)

    top_books = [prediction.iid for prediction in collaborative_predictions[:num_recommendations]]
    return Book.objects.filter(isbn__in=top_books)

def split_train_test_set(data, test_size=0.2):
    users_with_ratings = data['user_id'].unique()
    
    test_users = random.sample(list(users_with_ratings), int(len(users_with_ratings) * test_size))
    
    test_data = data[data['user_id'].isin(test_users)]
    
    train_data = data[~data['user_id'].isin(test_users)]
    
    reader = Reader(rating_scale=(1, 10))
    trainset = Dataset.load_from_df(train_data[['user_id', 'book__isbn', 'book_rating']], reader).build_full_trainset()
    testset = Dataset.load_from_df(test_data[['user_id', 'book__isbn', 'book_rating']], reader).build_full_trainset().build_testset()
    
    return trainset, testset

def evaluate_model(testset, svd):
    logger.info("Evaluating model...")
    predictions = [svd.predict(uid, iid).est for uid, iid, _ in testset]
    actuals = [true_r for _, _, true_r in testset]
    mse = mean_squared_error(actuals, predictions)
    logger.info(f"Evaluation completed with MSE: {mse}")
    print(f"Total Mean Squared Error for Hybrid Recommendations: {mse}")
    return mse

def evaluate_user_model(user_id, ratings_df, svd):
    logger.info(f"Evaluating model for user {user_id}...")
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    if user_ratings.empty:
        return None, "No ratings found for user"

    predictions = [svd.predict(user_id, isbn).est for isbn in user_ratings['book__isbn']]
    actuals = user_ratings['book_rating'].tolist()
    mse = mean_squared_error(actuals, predictions)
    
    logger.info(f"Evaluation completed with MSE: {mse} for user {user_id}")
    return mse, None

if __name__ == "__main__":
    # num_recommendations = 10
    # user_id = 239106 # Example user ID
    
    ratings_df = load_data()
    tfidf_matrix, books = build_tfidf_matrix()
    nn = load_or_compute_nn(tfidf_matrix)
    svd = load_or_compute_svd(ratings_df)
    
    # Commenting out user recommendation prints
    # content_recs = content_based_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, num_recommendations)
    # print("Content-Based Filtering Recommendations:")
    # print(content_recs)

    # collaborative_recs = collaborative_filtering_recommendations(user_id, ratings_df, svd, num_recommendations)
    # print("\nCollaborative Filtering Recommendations:")
    # for rec in collaborative_recs:
    #     print(rec.title)

    # hybrid_recs = hybrid_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, svd, num_recommendations)
    # print("\nHybrid Recommendations:")
    # for rec in hybrid_recs:
    #     print(rec.title)
    
    trainset, testset = split_train_test_set(ratings_df)
    
    mse_hybrid = evaluate_model(testset, svd)
    logger.info(f"\nMean Squared Error for Hybrid Recommendations: {mse_hybrid}")
