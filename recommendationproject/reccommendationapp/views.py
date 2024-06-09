import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD
from sklearn.metrics import mean_squared_error
from django.db.models import Avg
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.db.models import Count
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import User, Rating, Book
import requests

# Set the random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ratings data
def load_data():
    logger.info("Loading ratings data...")
    ratings = list(Rating.objects.values('user_id', 'book__isbn', 'book_rating'))
    logger.info("Ratings data loaded successfully.")
    return pd.DataFrame(ratings)

# Compute average rating for each book
def compute_average_ratings():
    logger.info("Computing average ratings for each book...")
    avg_ratings = Rating.objects.values('book__isbn').annotate(avg_rating=Avg('book_rating'))
    books = pd.DataFrame(list(Book.objects.all().values()))
    avg_ratings_df = pd.DataFrame(list(avg_ratings))
    books = books.merge(avg_ratings_df, left_on='isbn', right_on='book__isbn', how='left')
    books['avg_rating'].fillna(0, inplace=True)
    return books

# Build and cache the TF-IDF matrix
def build_tfidf_matrix():
    logger.info("Building TF-IDF matrix...")
    books = compute_average_ratings()
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
    svd = SVD(random_state=SEED)
    svd.fit(trainset)
    return svd

# Content-Based Filtering
def content_based_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, num_recommendations=10):
    logger.info(f"Generating content-based recommendations for user {user_id}...")
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]

    if user_id not in ratings_df['user_id'].values:
        return books.nlargest(num_recommendations, 'avg_rating')

    user_books_indices = [books.index[books['isbn'] == isbn].tolist()[0] for isbn in user_ratings['book__isbn']]
    sim_scores = nn.kneighbors(tfidf_matrix[user_books_indices], n_neighbors=len(books), return_distance=False).flatten()
    recommended_indices = [idx for idx in sim_scores if idx not in user_books_indices][:num_recommendations]
    return books.iloc[recommended_indices]

# Collaborative Filtering with SVD
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

# Hybrid Recommendation
def hybrid_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, svd, num_recommendations=10):
    logger.info(f"Generating hybrid recommendations for user {user_id}...")
    content_recs = content_based_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, num_recommendations * 2)
    content_rec_isbns = content_recs['isbn'].tolist()

    collaborative_predictions = [svd.predict(user_id, isbn) for isbn in content_rec_isbns]
    collaborative_predictions.sort(key=lambda x: x.est, reverse=True)

    top_books = [prediction.iid for prediction in collaborative_predictions[:num_recommendations]]
    return Book.objects.filter(isbn__in=top_books)

def get_random_user_ids():
    users_with_reviews = User.objects.annotate(num_reviews=Count('rating')).filter(num_reviews__gte=5)
    user_ids_with_reviews = list(users_with_reviews.values_list('user_id', flat=True))
    random.shuffle(user_ids_with_reviews)
    return user_ids_with_reviews[:10]  # Limit to 10 random user IDs

def homepage_view(request):
    random_user_ids = get_random_user_ids()
    return render(request, 'reccommendationapp/homepage.html', {
        'random_user_ids': random_user_ids,
        'selected_user_id': None,
    })

def user_ratings_view(request, user_id):
    random_user_ids = get_random_user_ids()
    user_ratings = None
    
    user = get_object_or_404(User, user_id=user_id)
    user_ratings = Rating.objects.filter(user=user).select_related('book')

    return render(request, 'reccommendationapp/user_ratings.html', {
        'user_id': user_id,
        'user_ratings': user_ratings,
        'random_user_ids': random_user_ids,
        'selected_user_id': user_id,
    })

def user_recommendations_view(request, user_id):
    random_user_ids = get_random_user_ids()
    # Make an internal API call to fetch recommendations
    api_url = request.build_absolute_uri(f"/api/fetch_hybrid_recommendations/{user_id}/")
    response = requests.get(api_url, headers={'Content-Type': 'application/json'})
    response_data = response.json()

    recommendations = response_data.get('recommendations', [])

    return render(request, 'reccommendationapp/user_recommendations.html', {
        'user_id': user_id,
        'recommendations': recommendations,
        'random_user_ids': random_user_ids,
        'selected_user_id': user_id,
    })

@api_view(['GET'])
def fetch_hybrid_recommendations(request, user_id):
    if not user_id:
        return Response({'error': 'User ID is required'}, status=400)
    
    user = get_object_or_404(User, user_id=user_id)

    # Fetch recommendations using the hybrid recommendation system
    logger.info("Loading ratings data...")
    ratings_df = load_data()  # Load ratings data consistently
    logger.info(f"Loaded ratings data with {len(ratings_df)} records")

    logger.info("Building TF-IDF matrix...")
    tfidf_matrix, books = build_tfidf_matrix()  # Build TF-IDF matrix
    logger.info(f"Built TF-IDF matrix for {len(books)} books")

    logger.info("Computing Nearest Neighbors model for books...")
    nn = load_or_compute_nn(tfidf_matrix)  # Compute Nearest Neighbors model

    logger.info("Computing SVD model for collaborative filtering...")
    svd = load_or_compute_svd(ratings_df)  # Compute SVD model

    logger.info("Generating hybrid recommendations...")
    recommended_books = hybrid_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, svd, num_recommendations=10)  # Generate hybrid recommendations

    recommendations = [
        {
            'title': book.title,
            'author': book.author,
            'isbn': book.isbn,
            'year_of_publication': book.year_of_publication,
            'image_url_m': book.image_url_m,
        }
        for book in recommended_books
    ]

    logger.info(f"Recommendations generated for user {user_id}: {[book['title'] for book in recommendations]}")

    return Response({'recommendations': recommendations})
