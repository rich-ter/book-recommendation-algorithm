from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, get_object_or_404
from .models import User, Rating, Book
from .forms import UserRecommendationForm
import random
from django.db.models import Count

def user_ratings_view(request):
    user_ratings = None
    recommendations = None
    user_id = None
    
    # Fetch user IDs of users who have at least 5 reviews
    users_with_reviews = User.objects.annotate(num_reviews=Count('rating')).filter(num_reviews__gte=5)
    user_ids_with_reviews = list(users_with_reviews.values_list('user_id', flat=True))
    random.shuffle(user_ids_with_reviews)
    random_user_ids = user_ids_with_reviews[:10]  # Limit to 10 random user IDs

    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        user = get_object_or_404(User, user_id=user_id)
        user_ratings = Rating.objects.filter(user=user).select_related('book')

        # Fetch recommendations (example logic, you might need to replace this with your actual recommendation algorithm)
        rated_books = user_ratings.values_list('book_id', flat=True)
        recommended_books = Book.objects.exclude(id__in=rated_books)[:10]  # Replace with your recommendation logic
        recommendations = [
            {
                'book': book,
                'predicted_rating': 'N/A'  # Replace with actual predicted rating logic
            }
            for book in recommended_books
        ]

    return render(request, 'reccommendationapp/user_ratings.html', {
        'user_id': user_id,
        'user_ratings': user_ratings,
        'recommendations': recommendations,
        'random_user_ids': random_user_ids,
    })