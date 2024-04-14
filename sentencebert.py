import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
# Check if the pickle file exists
pkl_file = 'movies_with_embeddings.pkl'
if os.path.exists(pkl_file):
    movies = pd.read_pickle(pkl_file)
else:
    # Load data
    movies = pd.read_csv('movies_metadata.csv')

    # Drop rows with missing values in 'overview' as we need descriptions to generate embeddings
    movies.dropna(subset=['overview'], inplace=True)

    # Simple preprocessing to ensure all texts are strings
    synopsis = movies['overview'].astype(str).values

    # Generate embeddings
    movies_embeddings = model.encode(synopsis, show_progress_bar=True)

    # Optionally, save these embeddings to disk to avoid recomputation
    movies['embeddings'] = movies_embeddings.tolist()
    movies.to_pickle(pkl_file)

def recommend_movies(user_input, movies_df, top_n=5):
    # Encode the user input
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    
    # Compute similarities
    movies_df['similarity'] = movies_df['embeddings'].apply(lambda emb: util.pytorch_cos_sim(user_embedding, torch.tensor(emb).to(device).unsqueeze(0)).item())
    
    # Find the top N movies with the highest similarity scores
    recommended_movies = movies_df.nlargest(top_n, 'similarity')
    
    return recommended_movies[['title', 'overview', 'similarity']]

user_input = input("Enter your movie preferences: ")
recommended_movies = recommend_movies(user_input, movies, top_n=5)
print(recommended_movies)
