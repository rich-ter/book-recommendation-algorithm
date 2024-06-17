import logging
import random
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.db.models import Count
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import User, Rating, Book
import requests
from .recommendation_algorithms import (
    load_data, compute_average_ratings, build_tfidf_matrix, load_or_compute_nn,
    load_or_compute_svd, content_based_recommendations, collaborative_filtering_recommendations,
    hybrid_recommendations, evaluate_user_model
)

SEED = 42
random.seed(SEED)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_random_user_ids():
    users_with_reviews = User.objects.annotate(num_reviews=Count('rating')).filter(num_reviews__gte=5)
    user_ids_with_reviews = list(users_with_reviews.values_list('user_id', flat=True))
    random.shuffle(user_ids_with_reviews)
    return user_ids_with_reviews[:10] 

def homepage_view(request):
    random_user_ids = get_random_user_ids()
    return render(request, 'reccommendationapp/homepage.html', {
        'random_user_ids': random_user_ids,
        'selected_user_id': None,
    })

def user_ratings_view(request, user_id):
    random_user_ids = get_random_user_ids()
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
    api_url = request.build_absolute_uri(f"/api/fetch_hybrid_recommendations/{user_id}/")
    
    try:
        response = requests.get(api_url, headers={'Content-Type': 'application/json'})
        response.raise_for_status()  
        response_data = response.json()

        recommendations = response_data.get('recommendations', [])
        mse = response_data.get('mse', None)
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        recommendations = []
        mse = None
    except ValueError as e:
        logger.error(f"JSON decoding failed: {e}")
        recommendations = []
        mse = None

    return render(request, 'reccommendationapp/user_recommendations.html', {
        'user_id': user_id,
        'recommendations': recommendations,
        'random_user_ids': random_user_ids,
        'selected_user_id': user_id,
        'mse': mse,
    })

@api_view(['GET'])
def fetch_hybrid_recommendations(request, user_id):
    if not user_id:
        return Response({'error': 'User ID is required'}, status=400)
    
    user = get_object_or_404(User, user_id=user_id)

    logger.info("Loading ratings data...")
    ratings_df = load_data() 
    logger.info(f"Loaded ratings data with {len(ratings_df)} records")

    logger.info("Building TF-IDF matrix...")
    tfidf_matrix, books = build_tfidf_matrix()  
    logger.info(f"Built TF-IDF matrix for {len(books)} books")

    logger.info("Computing Nearest Neighbors model for books...")
    nn = load_or_compute_nn(tfidf_matrix)  

    logger.info("Computing SVD model for collaborative filtering...")
    svd = load_or_compute_svd(ratings_df)  
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

    mse, error = evaluate_user_model(user_id, ratings_df, svd)
    if error:
        return Response({'error': error}, status=400)

    return Response({'recommendations': recommendations, 'mse': mse})
