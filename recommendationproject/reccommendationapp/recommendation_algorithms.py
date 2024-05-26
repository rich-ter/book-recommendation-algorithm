import pandas as pd
from .models import Book, Rating
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import NMF
import numpy as np

def load_data():
    books = pd.DataFrame(list(Book.objects.all().values()))
    ratings = pd.DataFrame(list(Rating.objects.all().values()))
    return books, ratings




def get_content_based_recommendations(user_id, books, ratings, num_recommendations=10):
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




def get_collaborative_recommendations(user_id, books, ratings, num_recommendations=10, n_components=20):
    user_item_matrix = ratings.pivot(index='user_id', columns='book_id', values='book_rating').fillna(0)
    if user_id not in user_item_matrix.index:
        return []
    nmf = NMF(n_components=n_components)
    user_factors = nmf.fit_transform(user_item_matrix)
    item_factors = nmf.components_
    user_idx = user_item_matrix.index.get_loc(user_id)
    user_vector = user_factors[user_idx]
    scores = np.dot(user_vector, item_factors)
    book_indices = np.argsort(scores)[::-1]
    user_rated_indices = user_item_matrix.columns[user_item_matrix.iloc[user_idx] > 0].tolist()
    recommended_indices = [idx for idx in book_indices if idx not in user_rated_indices][:num_recommendations]
    recommended_books = books[books['id'].isin(recommended_indices)]
    return recommended_books


def get_hybrid_recommendations(user_id, num_recommendations=10, content_weight=0.5, collab_weight=0.5):
    books, ratings = load_data()
    content_recommendations = get_content_based_recommendations(user_id, books, ratings, num_recommendations)
    collaborative_recommendations = get_collaborative_recommendations(user_id, books, ratings, num_recommendations)
    
    combined_recommendations = pd.concat([content_recommendations, collaborative_recommendations]).drop_duplicates()
    
    combined_recommendations['score'] = (
        combined_recommendations.index.map(lambda idx: content_weight * combined_recommendations.index.isin(content_recommendations.index).astype(int)[idx]) + 
        combined_recommendations.index.map(lambda idx: collab_weight * combined_recommendations.index.isin(collaborative_recommendations.index).astype(int)[idx])
    )
    
    combined_recommendations = combined_recommendations.sort_values(by='score', ascending=False)
    return combined_recommendations.head(num_recommendations)
