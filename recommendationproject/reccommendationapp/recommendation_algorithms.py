import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
from surprise import Dataset, Reader, SVD
from django.db import connection
from reccommendationapp.models import Book, Rating

# Content-Based Filtering
def content_based_recommendations(user_id, num_recommendations=10):
    books = pd.DataFrame(list(Book.objects.all().values()))
    ratings = pd.DataFrame(list(Rating.objects.all().values()))

    if user_id not in ratings['user_id'].values:
        # For cold start users, recommend popular books
        popular_books = books.nlargest(num_recommendations, 'average_rating')
        return popular_books

    user_ratings = ratings[ratings['user_id'] == user_id]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books['title'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    user_books_indices = [books.index[books['id'] == book_id].tolist()[0] for book_id in user_ratings['book_id']]
    sim_scores = cosine_sim[user_books_indices].sum(axis=0)
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in sim_scores if i[0] not in user_books_indices][:num_recommendations]
    recommended_books = books.iloc[recommended_indices]
    return recommended_books

# Collaborative Filtering with SVD
def collaborative_filtering_recommendations(user_id, num_recommendations=10):
    ratings_df = load_data()
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_df[['user_id', 'book_id', 'book_rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)

    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    if user_ratings.empty:
        # For cold start users, recommend popular books
        books = pd.DataFrame(list(Book.objects.all().values()))
        popular_books = books.nlargest(num_recommendations, 'average_rating')
        return popular_books

    all_books = ratings_df['book_id'].unique()
    unrated_books = [book for book in all_books if book not in user_ratings['book_id'].values]
    
    predictions = [svd.predict(user_id, book) for book in unrated_books]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:num_recommendations]
    
    top_books = [prediction.iid for prediction in top_predictions]
    recommended_books = Book.objects.filter(id__in=top_books)
    return recommended_books

# Hybrid Recommendation
def hybrid_recommendations(user_id, book_id, top_n=10):
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    with open('book_indices.pkl', 'rb') as f:
        book_indices = pickle.load(f)
    with open('svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)

    book_idx = book_indices.get(book_id)
    if book_idx is None:
        return []

    sim_scores = list(enumerate(cosine_sim[book_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    content_recommendations = [book_indices['id'][book_idx] for book_idx, _ in sim_scores[1:top_n + 1]]

    collaborative_recommendations = []
    for book_id in content_recommendations:
        predicted_rating = svd_model.predict(user_id, book_id).est
        collaborative_recommendations.append((book_id, predicted_rating))

    collaborative_recommendations = sorted(collaborative_recommendations, key=lambda x: x[1], reverse=True)
    top_books = [rec[0] for rec in collaborative_recommendations[:top_n]]
    recommended_books = Book.objects.filter(id__in=top_books)
    return recommended_books

def load_data():
    with connection.cursor() as cursor:
        cursor.execute("SELECT user_id, book_id, book_rating FROM reccommendationapp_rating")
        ratings = cursor.fetchall()
    ratings_df = pd.DataFrame(ratings, columns=['user_id', 'book_id', 'book_rating'])
    return ratings_df


