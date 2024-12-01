import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
movies_metadata = pd.read_csv('C:\\Users\\schen\\software-design\\MovieRepo-COIL\\tmdb_5000_movies.csv')
credits = pd.read_csv('C:\\Users\\schen\\software-design\\MovieRepo-COIL\\tmdb_5000_credits.csv')

# Convert the 'id' and 'movie_id' columns to strings, filling NaN values with a placeholder like an empty string
movies_metadata['id'] = movies_metadata['id'].astype(str).fillna('')
credits['movie_id'] = credits['movie_id'].astype(str).fillna('')

# Merge the datasets on the 'movie_id' column
movies = pd.merge(movies_metadata, credits, left_on='id', right_on='movie_id')

# Check the data
print(movies.head())
print(movies.columns)
'''
print(movies_metadata['id'].isnull().sum())  # Should be 0 if there are no missing values
print(credits['movie_id'].isnull().sum()) 
print(movies_metadata[movies_metadata['id'].str.isnumeric() == False]['id'].unique())
print(credits[credits['movie_id'].str.isnumeric() == False]['movie_id'].unique())
'''

# Select important columns
movies = movies[["movie_id", 'original_title', 'overview', 'genres', 'cast', 'crew', 'vote_average']]

# Handle missing values (e.g., filling NaNs with an empty string)
movies['overview'] = movies['overview'].fillna('')

# Convert 'genres', 'cast', and 'crew' from stringified lists to actual lists
movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if isinstance(x, str) else [])
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]] if isinstance(x, str) else [])  # Get top 3 cast members
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'] if isinstance(x, str) else [])