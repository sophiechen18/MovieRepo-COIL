import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_metadata = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies_metadata['id'] = movies_metadata['id'].astype(str).fillna('')
credits['movie_id'] = credits['movie_id'].astype(str).fillna('')

# merge the datasets
movies = pd.merge(movies_metadata, credits, left_on='id', right_on='movie_id')

movies = movies[["movie_id", 'original_title', 'overview', 'genres', 'cast', 'crew', 'vote_average']]
print(movies['original_title'])

movies['overview'] = movies['overview'].fillna('')

movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if isinstance(x, str) else [])
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]] if isinstance(x, str) else [])  # Get top 3 cast members
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'] if isinstance(x, str) else [])

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    title = title.lower()

    idx = movies[movies['original_title'].str.lower() == title].index[0]  # get the index of the movie
    sim_scores = list(enumerate(cosine_sim[idx]))    # get similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # sort by similarity
    sim_scores = sim_scores[1:num_recommendations + 1]  # get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]  # get the indices of these movies

    if show_genres:
        recommended_movies = movies[['original_title', 'genres', 'overview', 'vote_average']].iloc[movie_indices].to_dict('records')
    else:
        recommended_movies = movies[['original_title', 'overview', 'vote_average']].iloc[movie_indices].to_dict('records')
    
    # return the titles and additional info (e.g., genre, vote average)
    return recommended_movies

# Streamlit app

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        font-family: monospace;
        color: ##A30B0B; /* Optional: Customize the color */
    }
    </style>
    <h1 class="title">Movie Recommendation System</h1>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Settings")
    num_recommendations = st.slider("Number of Recommendations", 5, 20, 10)  # default suggestion number is 10
    show_genres = st.checkbox("Show Genres", value=True)
    
col1, col2 = st.columns(2)

# user input 
with col1:
    st.text("Enter a movie title you would like similar recommendations for.")
    movie = st.text_input("Movie Title")

with col2:
    st.subheader("Results")
    if movie:
        try:
            recommendations = get_recommendations(movie)
            for rec in recommendations:
                st.write(rec)
        except IndexError:
            st.error("Movie not found.")


