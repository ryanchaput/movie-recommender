import pandas as pd
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the data
data = pd.read_csv("mpst_full_data.csv", sep=",")
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def preprocess_text(text):
    # Tokenization, stop word removal, and stemming
    tokens = text.lower().split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

preprocessed_file = "preprocessed_data.csv"

try:
    # Try to load the preprocessed data from file
    preprocessed_data = pd.read_csv(preprocessed_file)
except FileNotFoundError:
    # If the file doesn't exist, preprocess the data and save it to file
    preprocessed_data = data[["imdb_id", "plot_synopsis"]].copy()
    preprocessed_data["preprocessed_plot"] = preprocessed_data["plot_synopsis"].apply(preprocess_text)
    preprocessed_data[["imdb_id", "preprocessed_plot"]].to_csv(preprocessed_file, index=False)

# Merge the preprocessed data with the original dataframe
data = pd.merge(data, preprocessed_data[["imdb_id", "preprocessed_plot"]], on="imdb_id")

# Vectorize the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["preprocessed_plot"])
y = data["tags"]

# Train the Bayesian model
model = MultinomialNB()
model.fit(X, y)

# Process user input
user_input = preprocess_text(input("Enter your movie preferences: "))
user_input_vector = vectorizer.transform([user_input])

# Make recommendations
predicted_probabilities = model.predict_proba(user_input_vector)[0]
predicted_labels = model.classes_
sorted_indices = predicted_probabilities.argsort()[::-1]

# Print the recommendations with probabilities
print("Recommended Movies:")
for index in sorted_indices[:5]:
    label = predicted_labels[index]
    probability = predicted_probabilities[index]
    recommended_movies = data[data["tags"] == label]
    similarity_scores = cosine_similarity(user_input_vector, vectorizer.transform(recommended_movies["preprocessed_plot"]))[0]
    top_indices = similarity_scores.argsort()[-1:][::-1]
    recommended_movie = recommended_movies.iloc[top_indices]
    print(f"Movie: {recommended_movie['title'].values[0]}")
    print(f"Probability: {probability:.2f}")
    print(f"Plot Synopsis: {recommended_movie['plot_synopsis'].values[0]}")
    print()
