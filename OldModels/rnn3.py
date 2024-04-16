import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the merged CSV file into a DataFrame
df = pd.read_csv('merged_file.csv')

# Preprocess the data
df['text'] = df['overview'].astype(str) + ' ' + df['genres'].astype(str) + ' ' + df['keywords'].astype(str)

# Split the data into training and testing sets
X_train, X_test, _, _ = train_test_split(df['text'], df['title'], test_size=0.2, random_state=42)

# Load the pre-trained SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Check if the encoded embeddings file exists
if os.path.exists('rnn_encoded_embeddings.npz'):
    # Load the encoded embeddings from the file
    with np.load('rnn_encoded_embeddings.npz') as data:
        X_train_embeddings = data['train_embeddings']
        X_test_embeddings = data['test_embeddings']
else:
    # Encode the movie overviews using SBERT
    X_train_embeddings = sbert_model.encode(X_train.tolist())
    X_test_embeddings = sbert_model.encode(X_test.tolist())
    
    # Save the encoded embeddings to a file
    np.savez('rnn_encoded_embeddings.npz', train_embeddings=X_train_embeddings, test_embeddings=X_test_embeddings)

# Reshape the embeddings to match the expected input shape of the LSTM layer
X_train_reshaped = X_train_embeddings.reshape(X_train_embeddings.shape[0], 1, X_train_embeddings.shape[1])
X_test_reshaped = X_test_embeddings.reshape(X_test_embeddings.shape[0], 1, X_test_embeddings.shape[1])

# Build the RNN model
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(1, 384)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(X_train_embeddings.shape[1], activation='linear'))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.1, mode="min", baseline=None)
model.fit(X_train_reshaped, X_train_embeddings, validation_data=(X_test_reshaped, X_test_embeddings), epochs=10, batch_size=32, callbacks=[early_stop])

# Make recommendations
def recommend_movies(user_input, top_n=5):
    input_embedding = sbert_model.encode([user_input])
    input_reshaped = input_embedding.reshape(1, 1, input_embedding.shape[1])
    predicted_embedding = model.predict(input_reshaped)
    
    # Calculate cosine similarity between the predicted embedding and movie embeddings
    similarities = cosine_similarity(predicted_embedding, X_test_embeddings)
    movie_indices = similarities.argsort()[0][-top_n:][::-1]

    # Get the recommended movies (title, overview, probability)
    recommended_movies = df.iloc[X_test.index[movie_indices]][['title', 'overview']]
    recommended_movies['probability'] = similarities[0][movie_indices]
    
    return recommended_movies

# Get user input and recommend movies until the user quits
while True:
    user_input = input('Enter a movie description (or "quit" to exit): ')
    if user_input.lower() == 'quit':
        break
    recommended_movies = recommend_movies(user_input)
    print(recommended_movies)
    print('\n')
