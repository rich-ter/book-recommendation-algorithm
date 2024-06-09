from django.urls import path
from .views import homepage_view, user_ratings_view, fetch_hybrid_recommendations, user_recommendations_view

urlpatterns = [
    path('', homepage_view, name='homepage'),
    path('ratings/<int:user_id>/', user_ratings_view, name='user_ratings'),
    path('recommendations/<int:user_id>/', user_recommendations_view, name='user_recommendations'),
    path('api/fetch_hybrid_recommendations/<int:user_id>/', fetch_hybrid_recommendations, name='fetch_hybrid_recommendations'),
]
