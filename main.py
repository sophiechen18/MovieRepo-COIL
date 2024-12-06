import streamlit as st
import pandas as pd
import requests
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_API_KEY = "da7812d5a36a96ec885b30dd3fcffe79"
BASE_POSTER_URL = "https://image.tmdb.org/t/p/w500"

# Load the datasets
movies_metadata = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Convert the 'id' and 'movie_id' columns to strings, filling NaN values with a placeholder like an empty string
movies_metadata['id'] = movies_metadata['id'].astype(str).fillna('')
credits['movie_id'] = credits['movie_id'].astype(str).fillna('')

# Merge the datasets on the 'movie_id' column
movies = pd.merge(movies_metadata, credits, left_on='id', right_on='movie_id')

# Select important columns
movies = movies[["movie_id", 'original_title', 'overview', 'genres', 'cast', 'crew', 'vote_average']]
print(movies['original_title'])

# Handle missing values (e.g., filling NaNs with an empty string)
movies['overview'] = movies['overview'].fillna('')

# Convert 'genres', 'cast', and 'crew' from stringified lists to actual lists
movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if isinstance(x, str) else [])
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]] if isinstance(x, str) else [])  # Get top 3 cast members
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'] if isinstance(x, str) else [])

# Create the TF-IDF matrix for 'overview'
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    title = title.lower()

    idx = movies[movies['original_title'].str.lower() == title].index[0]  # Get the index of the movie
    sim_scores = list(enumerate(cosine_sim[idx]))    # Get similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity
    sim_scores = sim_scores[1:num_recommendations + 1]  # Get the top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]  # Get the indices of these movies

    if show_genres:
        recommended_movies = movies[['original_title', 'genres', 'overview', 'vote_average']].iloc[movie_indices].to_dict('records')
    else:
        recommended_movies = movies[['original_title', 'overview', 'vote_average']].iloc[movie_indices].to_dict('records')
    
    # Return the titles and additional info (e.g., genre, vote average)
    return recommended_movies

def fetch_poster_from_tmdb(movie_title):
    """
    Fetch the movie poster URL from TMDB API using the movie title.
    """
    search_url = f"https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path', None)
            if poster_path:
                return f"{BASE_POSTER_URL}{poster_path}"
    return "https://via.placeholder.com/500x750?text=No+Image"  # Placeholder for missing images

# Streamlit app

page = st.sidebar.radio("Navigation", ["Home", "Recommendations"])

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        font-family: monospace;
        color: #63bef2;
        background-color: white;
    }
    .movie-container {
        border: 5px solid #ddd;
        border-radius: 8px;
        border-color: #63bef2;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if page == "Home":
    # Home Page: Display popular movies
    st.title("Welcome to the Movie Recommendation System")
    st.subheader("Popular Movies")
    popular_movies = [
        "Inception", "The Dark Knight", "Avatar", "Interstellar", "Titanic", 
        "Inside Out", "Iron Man", "Frozen", "The Avengers", "The Matrix", 
        "Pulp Fiction", "Forrest Gump", "The Lion King", "Toy Story", "Shrek"
    ]

    num_columns = 5
    rows = [popular_movies[i:i + num_columns] for i in range(0, len(popular_movies), num_columns)]

    for row in rows:
        cols = st.columns(num_columns)
        for col, title in zip(cols, row):
            poster_url = fetch_poster_from_tmdb(title)
            col.image(poster_url, caption=title, use_container_width=True)

# Input from the user

elif page == "Recommendations":
    with st.sidebar:
        st.header("Settings")
        num_recommendations = st.slider("Number of Recommendations", 5, 20, 10)  # Default is 10
        show_genres = st.checkbox("Show Genres", value=True)
    
    # Recommendations Page
    st.title("Movie Recommendations")
    st.subheader("Enter a Movie Title")
    movie = st.text_input("Movie Title")

    if movie:
        try:
            recommendations = get_recommendations(movie)
            for rec in recommendations:
                poster_url = fetch_poster_from_tmdb(rec['original_title'])
                st.markdown(
                    f"""
                    <div class="movie-container">
                        <img src="{poster_url}" style="width:200px; float:left; margin-right:20px; border-radius:8px;">
                        <h3 style="margin-bottom: 5px;"><strong>🎬 {rec['original_title']}</strong></h3>
                        <p>{"<b>Genres:</b> " + ", ".join(rec['genres']) if show_genres else ""}</p>
                        <p><b>Overview:</b> {rec['overview']}</p>
                        <p><b>Rating:</b> ⭐ {rec['vote_average']}</p>
                        <div style="clear:both;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except IndexError:
            st.error("Movie not found in dataset.")
    else:
        st.info("Enter a movie title to get recommendations.")
