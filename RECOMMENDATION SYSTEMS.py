import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies_data = {
    'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
    'genre': ['Drama', 'Crime, Drama', 'Action, Crime, Drama', 'Crime, Drama', 'Drama, Romance'],
    'director': ['Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan', 'Quentin Tarantino', 'Robert Zemeckis'],
    'actors': ['Tim Robbins, Morgan Freeman', 'Marlon Brando, Al Pacino', 'Christian Bale, Heath Ledger', 'John Travolta, Uma Thurman', 'Tom Hanks, Robin Wright'],
    'plot_keywords': ['imprisonment, escape', 'mafia, crime', 'dc comics, joker', 'violence, black comedy', 'slow motion scene, vietnam war']
}

movies_df = pd.DataFrame(movies_data)

# Combine text data for each movie
movies_df['combined_features'] = movies_df['genre'] + ' ' + movies_df['director'] + ' ' + movies_df['actors'] + ' ' + movies_df['plot_keywords']

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])

# Calculate cosine similarity between movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Function to recommend movies based on user preferences
def recommend_movies(user_preferences, top_n=5):
    # Transform user preferences into TF-IDF vector
    user_tfidf = tfidf_vectorizer.transform([user_preferences])

    # Calculate similarity scores between user preferences and all movies
    similarity_scores = linear_kernel(user_tfidf, tfidf_matrix).flatten()

    # Get indices of movies with highest similarity scores
    top_indices = similarity_scores.argsort()[:-top_n-1:-1]

    # Return recommended movies
    recommended_movies = [(movies_df['title'][i], similarity_scores[i]) for i in top_indices]
    return recommended_movies

# Example usage
user_preferences = 'crime drama'
recommended_movies = recommend_movies(user_preferences)
print("Recommended movies based on your preferences:")
for movie, score in recommended_movies:
    print(f"{movie} (Score: {score:.2f})")