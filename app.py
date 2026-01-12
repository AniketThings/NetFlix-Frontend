from flask import Flask, render_template, request
import os
import pandas as pd
from pymongo import MongoClient
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# MongoDB connection (environment-based)
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise Exception("MONGO_URI environment variable not set")

client = MongoClient(mongo_uri)
db = client.NoOneFlix
movies_collection = db.movies
ratings_collection = db.ratings

# Fetch data
movies = pd.DataFrame(list(movies_collection.find()))
ratings = pd.DataFrame(list(ratings_collection.find()))

if movies.empty or ratings.empty:
    print("MongoDB collections are empty. App started without recommendation data.")

# Prepare model only if data exists
if not movies.empty and not ratings.empty:
    movies_with_ratings = movies.merge(ratings, on='movieId')
    pivot_table = movies_with_ratings.pivot_table(
        index='title', columns='userId', values='rating'
    ).fillna(0)

    sparse_matrix = csr_matrix(pivot_table)
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(sparse_matrix)
else:
    pivot_table = None
    model = None

def recommend_movies_by_id(movie_id):
    if model is None:
        return ["Recommendation system not initialized (no data)."]

    try:
        movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
        movie_idx = pivot_table.index.get_loc(movie_title)

        distances, suggestions = model.kneighbors(
            pivot_table.iloc[movie_idx, :].values.reshape(1, -1),
            n_neighbors=6
        )
        return [pivot_table.index[i] for i in suggestions[0][1:]]
    except Exception as e:
        return [f"Error: {str(e)}"]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('movie-id-input.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_id = request.form.get('movieId', type=int)
    if movie_id is None:
        return render_template('result.html', recommendations=["Invalid Movie ID."])

    recommendations = recommend_movies_by_id(movie_id)
    return render_template('result.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

