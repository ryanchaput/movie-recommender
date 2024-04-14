import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the movies dataset
data = pd.read_csv('movies_metadata.csv', low_memory=False)
data = data[['id', 'title', 'overview', 'genres']]  # Assuming genres are used for tags

# Drop rows where overview is NaN
data.dropna(subset=['overview'], inplace=True)

# Simplify genres into a single column if necessary
data['genres'] = data['genres'].apply(lambda x: eval(x)[0]['name'] if x != '[]' else None)
data.dropna(subset=['genres'], inplace=True)

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the overview text
data['embeddings'] = data['overview'].apply(lambda x: model.encode(x))

# Since embeddings are high-dimensional, reduce them using PCA for Naive Bayes
pca = PCA(n_components=50)  # Adjust components based on variance explained
X = pca.fit_transform(list(data['embeddings']))
y = data['genres'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = gnb.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

def recommend_movie(user_input):
    # Process and encode the user input
    user_input_embedding = model.encode(user_input)
    user_input_pca = pca.transform([user_input_embedding])

    # Predict the genre
    predicted_genre = gnb.predict(user_input_pca)[0]

    # Filter movies by the predicted genre and recommend one
    genre_movies = data[data['genres'] == predicted_genre]
    recommended_movie = genre_movies.sample(1)  # Randomly pick one movie from the genre

    return recommended_movie[['title', 'overview']]

# Get user input and recommend a movie
user_input = input("Enter your movie preferences: ")
recommended_movie = recommend_movie(user_input)
print(recommended_movie)
