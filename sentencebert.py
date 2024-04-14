import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load data
movies = pd.read_csv('movies_metadata.csv', usecols=['title', 'overview'])

# Drop rows with missing values in 'overview' as we need descriptions to generate embeddings
movies.dropna(subset=['overview'], inplace=True)

# Simple preprocessing to ensure all texts are strings
synopsis = movies['overview'].astype(str).values
model = SentenceTransformer('all-MiniLM-L6-v2')
# Generate embeddings
movies['embeddings'] = model.encode(synopsis, show_progress_bar=True)

# Optionally, save these embeddings to disk to avoid recomputation
movies.to_pickle('movies_with_embeddings.pkl')
def recommend_movies(user_input, movies_df, top_n=5):
    # Encode the user input
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    
    # Compute similarities
    movies_df['similarity'] = movies_df['embeddings'].apply(lambda emb: util.pytorch_cos_sim(user_embedding, emb).item())
    
    # Find the top N movies with the highest similarity scores
    recommended_movies = movies_df.nlargest(top_n, 'similarity')
    
    return recommended_movies[['title', 'overview', 'similarity']]
user_input = input("Enter your movie preferences: ")
recommended_movies = recommend_movies(user_input, movies, top_n=5)
print(recommended_movies)
