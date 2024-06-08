from django.urls import path
from .views import user_ratings_view

urlpatterns = [
    path('', user_ratings_view, name='user_ratings'),
]
