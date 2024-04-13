from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the cleaned movie dataset
df = pd.read_csv('cleaned_file.csv')

# Load the SentenceTransformer model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Generate embeddings for movie plot synopses
movie_synopses = df['overview'].astype(str).values
titles = df['title'].astype(str).values

movie_embeddings = model.encode(movie_synopses)

# User input
user_input = input("Enter your movie preferences: ")

# Generate embedding for user input
user_embedding = model.encode([user_input])

# Calculate cosine similarity between user input and movie embeddings
similarity_scores = cosine_similarity(user_embedding, movie_embeddings)

# Get top-N recommended movies
top_n = 5
top_indices = similarity_scores.argsort()[0][-top_n:][::-1]
recommended_movies = [(titles[i], movie_synopses[i]) for i in top_indices]

# Present the recommended movies
for title, synopsis in recommended_movies:
    print("Title:", title)
    print()
    print("Synopsis:", synopsis)
    print()