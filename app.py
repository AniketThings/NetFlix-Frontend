from flask import Flask, render_template, request
from pymongo import MongoClient
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client.NoOneFlix
movies_collection = db.movies
ratings_collection = db.ratings

# Fetch data from MongoDB
movies = pd.DataFrame(list(movies_collection.find()))
ratings = pd.DataFrame(list(ratings_collection.find()))

# Merge movies and ratings
movies_with_ratings = movies.merge(ratings, on='movieId')

# Create pivot table
pivot_table = movies_with_ratings.pivot_table(index='title', columns='userId', values='rating').fillna(0)

# Create sparse matrix
sparse_matrix = csr_matrix(pivot_table)

# Train KNN model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_matrix)

def recommend_movies_by_id(movie_id):
    try:
        # Get movie title for the given movieId
        movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
        movie_idx = pivot_table.index.get_loc(movie_title)
        
        # Find similar movies
        distances, suggestions = model.kneighbors(
            pivot_table.iloc[movie_idx, :].values.reshape(1, -1), 
            n_neighbors=6
        )
        # Return the recommended movie titles (skip the first as it's the queried movie)
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
    app.run(debug=True)











