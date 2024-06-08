from django.urls import path
from .views import user_ratings_view, fetch_recommendations

urlpatterns = [
    path('', user_ratings_view, name='user_ratings'),
    path('fetch_recommendations/', fetch_recommendations, name='fetch_recommendations'),

]
