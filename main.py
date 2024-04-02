import pyro
import pyro.distributions as dist
import torch
import numpy as np
import pandas as pd
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Read the dataset from the CSV file
data = pd.read_csv("spotify_songs.csv", sep=",")

# Preprocess the data
data["track_popularity"] = data["track_popularity"].astype(int)
data["duration_ms"] = data["duration_ms"].astype(int)

# Convert the DataFrame to a list of dictionaries
songs = data.to_dict(orient="records")

# Select only the relevant numeric columns for the model
model_data = data[["track_popularity", "danceability", "energy"]]

# Define the Bayesian model
def model(song_data):
    # Prior distributions for model parameters
    popularity_mean = pyro.sample("popularity_mean", dist.Normal(0., 10.))
    popularity_std = pyro.sample("popularity_std", dist.HalfNormal(10.))
    
    danceability_mean = pyro.sample("danceability_mean", dist.Normal(0., 1.))
    danceability_std = pyro.sample("danceability_std", dist.HalfNormal(1.))
    
    energy_mean = pyro.sample("energy_mean", dist.Normal(0., 1.))
    energy_std = pyro.sample("energy_std", dist.HalfNormal(1.))
    
    # Likelihood function
    with pyro.plate("data", len(song_data)):
        popularity = pyro.sample("popularity", dist.Normal(popularity_mean, popularity_std), obs=song_data[:, 0])
        danceability = pyro.sample("danceability", dist.Normal(danceability_mean, danceability_std), obs=song_data[:, 1])
        energy = pyro.sample("energy", dist.Normal(energy_mean, energy_std), obs=song_data[:, 2])

# Function to extract features from user input
def extract_features(user_input):
    doc = nlp(user_input)
    features = {}
    for token in doc:
        if token.text.lower() in ["popular", "popularity"]:
            features["track_popularity"] = 70
        elif token.text.lower() in ["danceable", "danceability"]:
            features["danceability"] = 0.7
        elif token.text.lower() in ["energetic", "energy"]:
            features["energy"] = 0.8
    return features

# Function to calculate the probability of a song being relevant to user preferences
def calculate_song_probability(song, user_features, posterior_samples):
    song_probs = []
    for feature, value in user_features.items():
        if feature == "track_popularity":
            feature_mean = posterior_samples["popularity_mean"].mean()
            feature_std = posterior_samples["popularity_std"].mean()
        elif feature == "danceability":
            feature_mean = posterior_samples["danceability_mean"].mean()
            feature_std = posterior_samples["danceability_std"].mean()
        elif feature == "energy":
            feature_mean = posterior_samples["energy_mean"].mean()
            feature_std = posterior_samples["energy_std"].mean()
        else:
            continue
        song_prob = torch.exp(-0.5 * ((song[feature] - feature_mean) / feature_std) ** 2)
        song_probs.append(song_prob.item())
    return np.mean(song_probs)

# Function to recommend songs based on user input
def recommend_songs(user_input, posterior_samples, top_n=5):
    user_features = extract_features(user_input)
    song_probs = []
    for song in songs:
        song_prob = calculate_song_probability(song, user_features, posterior_samples)
        song_probs.append(song_prob)
    top_songs = sorted(zip(songs, song_probs), key=lambda x: x[1], reverse=True)[:top_n]
    return top_songs

# Perform inference
from pyro.infer import MCMC, NUTS

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(torch.tensor(model_data.values, dtype=torch.float32))

posterior_samples = mcmc.get_samples()


# Example usage
user_input = input("Describe the kind of songs you want to listen to: ")
recommended_songs = recommend_songs(user_input, posterior_samples)
print("Recommended Songs:")
for song, prob in recommended_songs:
    print(f"{song['track_name']} by {song['track_artist']} (Probability: {prob})")