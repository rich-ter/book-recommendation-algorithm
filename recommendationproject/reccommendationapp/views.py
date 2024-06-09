import os
import sys
import json
import time
import logging
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .models import User, Rating, Book
from django.db.models import Count
import random

# Ensure the recommendation module is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maybe_solution import load_data, build_tfidf_matrix, load_or_compute_nn, load_or_compute_svd, hybrid_recommendations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def user_ratings_view(request):
    user_ratings = None
    user_id = None
    
    # Fetch user IDs of users who have at least 5 reviews
    users_with_reviews = User.objects.annotate(num_reviews=Count('rating')).filter(num_reviews__gte=5)
    user_ids_with_reviews = list(users_with_reviews.values_list('user_id', flat=True))
    random.shuffle(user_ids_with_reviews)
    random_user_ids = user_ids_with_reviews[:10]  # Limit to 10 random user IDs

    if request.method == 'POST' and request.POST.get('action_type') == 'load_ratings':
        user_id = request.POST.get('user_id')
        user = get_object_or_404(User, user_id=user_id)
        user_ratings = Rating.objects.filter(user=user).select_related('book')

    return render(request, 'reccommendationapp/user_ratings.html', {
        'user_id': user_id,
        'user_ratings': user_ratings,
        'random_user_ids': random_user_ids,
    })

def fetch_recommendations(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_id = data.get('user_id')
        
        if not user_id:
            return JsonResponse({'error': 'User ID is required.'}, status=400)
        
        user = get_object_or_404(User, user_id=user_id)

        # Fetch recommendations using the hybrid recommendation system
        logger.info("Loading ratings data...")
        ratings_df = load_data()
        logger.info("Building TF-IDF matrix...")
        tfidf_matrix, books = build_tfidf_matrix()
        logger.info("Computing Nearest Neighbors model for books...")
        nn = load_or_compute_nn(tfidf_matrix)
        logger.info("Computing SVD model for collaborative filtering...")
        svd = load_or_compute_svd(ratings_df)
        logger.info("Generating hybrid recommendations...")
        start_time = time.time()
        recommended_books = hybrid_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, svd, num_recommendations=10)
        end_time = time.time()

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
        
        logger.info(f"Total time taken: {end_time - start_time} seconds")
        return JsonResponse({'recommendations': recommendations})
    
    return JsonResponse({'error': 'Invalid request method.'}, status=405)
