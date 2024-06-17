# Book Recommendation System

<!-- ![Homepage](./project_images/homepage.png) -->

## Overview

The Book Recommendation System is a web-based application that leverages both content-based and collaborative filtering techniques to provide personalized book recommendations to users. Built with Django, this system integrates robust machine learning algorithms to analyze user preferences and deliver tailored suggestions.

## Features

- **Content-Based Filtering:** Uses TF-IDF vectorization to analyze book descriptions and recommend books with similar content.
- **Collaborative Filtering:** Utilizes matrix factorization (SVD) to identify user preferences based on ratings and recommend books liked by similar users.
- **Hybrid Recommendations:** Combines both content-based and collaborative filtering methods to improve recommendation accuracy.
- **User Interaction:** Allows users to view personalized recommendations based on their ratings and interactions.
- **API Integration:** Provides an API endpoint for accessing recommendation results programmatically.
- **Mass Import and Preprocessing:** Custom management commands for efficient data import and preprocessing.

## Technology Stack

- **Backend:** Django, Django REST Framework
- **Machine Learning:** scikit-learn, Surprise library
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite (default with Django)

## Key Components

### Database Design

The database schema includes models for books, users, and ratings, ensuring efficient data management and retrieval.

![Database Schema](./project_images/db-schema.png)

### Content-Based Filtering

Content-based filtering analyzes book descriptions using TF-IDF vectorization to recommend books with similar content to those the user has rated highly. This approach ensures that recommendations are based on the intrinsic properties of the books.

<!-- ![Content-Based Recommendations](./project_images/recommendations.png) -->

### Collaborative Filtering

Collaborative filtering uses matrix factorization (SVD) to recommend books based on user rating patterns, finding similarities between users. This approach leverages user behavior to predict preferences.

### Hybrid Model

The hybrid model combines content-based and collaborative filtering techniques, enhancing recommendation accuracy by leveraging the strengths of both methods. This approach provides a balanced recommendation that considers both book content and user behavior.

### User Interface

#### Homepage

The homepage displays random user IDs with their respective ratings, allowing users to explore and interact with the recommendation system.

![Homepage](./project_images/homepage.png)

#### User Ratings

Users can view and rate books, which are then used to generate personalized recommendations.

![User Ratings](./project_images/ratings.png)

#### Recommendations

Users receive tailored book recommendations based on their ratings and interactions with the system.

![Recommendations](./project_images/recommendations.png)

### API

The system includes an API for programmatic access to the recommendation results, allowing integration with other applications.

![API](./project_images/api.png)

### Mass Import and Preprocessing

The system supports mass import and preprocessing of data through custom management commands. This feature ensures efficient handling of large datasets and prepares the data for analysis and model training.

## Evaluation

The system's performance is evaluated using Mean Squared Error (MSE), with results indicating the accuracy of the recommendations:

- **Content-Based MSE:** Computed for content-based recommendations.
- **Collaborative-Based MSE:** Computed for collaborative filtering recommendations.
- **Hybrid MSE:** Computed for the combined recommendations.

![Model Output](./project_images/model_output.png)

## Machine Learning Implementation

### Content-Based Filtering

Content-based filtering analyzes book descriptions to recommend books with similar content. This method uses the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert book descriptions into numerical vectors. The cosine similarity between these vectors is then calculated to identify books that are similar to those a user has rated highly. 

Key steps:
1. **TF-IDF Vectorization:** Convert book descriptions into numerical vectors using the TF-IDF vectorizer.
2. **User Profile Creation:** Aggregate the TF-IDF vectors of books the user has rated positively to create a user profile.
3. **Similarity Calculation:** Compute the cosine similarity between the user profile and all book vectors to find the most similar books.

```python
def build_tfidf_matrix():
    books = compute_average_ratings()
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(books['title'])
    return tfidf_matrix, books

def content_based_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, num_recommendations=10):
    user_profile = create_user_profile(user_id, ratings_df, tfidf_matrix, books)
    sim_scores = nn.kneighbors(user_profile, n_neighbors=len(books), return_distance=False).flatten()
    recommended_indices = [idx for idx in sim_scores if idx not in user_books_indices][:num_recommendations]
    return books.iloc[recommended_indices]
```

### Collaborative Filtering

Collaborative filtering leverages the collective preferences of all users to make recommendations. This method uses Singular Value Decomposition (SVD) to decompose the user-item interaction matrix into latent factors. These factors are then used to predict a user's rating for unrated books, helping identify books that similar users have liked.

Key steps:
1. **Matrix Factorization:** Decompose the user-item interaction matrix using SVD to identify latent factors representing user preferences and item attributes.
2. **Prediction:** Use these latent factors to predict ratings for unrated books.
3. **Recommendation:** Recommend books with the highest predicted ratings.

```python
def load_or_compute_svd(ratings_df):
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_df[['user_id', 'book__isbn', 'book_rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD(random_state=SEED)
    svd.fit(trainset)
    return svd

def collaborative_filtering_recommendations(user_id, ratings_df, svd, num_recommendations=10):
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    unrated_books = [isbn for isbn in all_books if isbn not in user_ratings['book__isbn'].values]
    predictions = [svd.predict(user_id, isbn) for isbn in unrated_books]
    top_books = [prediction.iid for prediction in predictions[:num_recommendations]]
    return Book.objects.filter(isbn__in=top_books)
```

### Hybrid Recommendations

The hybrid recommendation approach combines both content-based and collaborative filtering methods to enhance recommendation accuracy. This approach first generates content-based recommendations and then refines these recommendations using collaborative filtering predictions.

Key steps:
1. **Content-Based Recommendations:** Generate an initial set of recommendations based on content similarity.
2. **Collaborative Filtering Adjustment:** Refine these recommendations using collaborative filtering to ensure they align with the broader user preferences captured by the collaborative model.

```python
def hybrid_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, svd, num_recommendations=10):
    content_recs = content_based_recommendations(user_id, ratings_df, tfidf_matrix, books, nn, num_recommendations * 2)
    content_rec_isbns = content_recs['isbn'].tolist()
    collaborative_predictions = [svd.predict(user_id, isbn) for isbn in content_rec_isbns]
    top_books = [prediction.iid for prediction in collaborative_predictions[:num_recommendations]]
    return Book.objects.filter(isbn__in=top_books)
```

## Conclusion

The Book Recommendation System is a comprehensive solution for personalized book recommendations, integrating advanced machine learning techniques and a user-friendly web interface. It demonstrates practical skills in backend development, machine learning, and frontend design, making it an attractive project for potential employers.
