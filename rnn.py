import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

movie_synopses = [...]  # List of movie synopses
release_dates = [...]   # List of release dates 
genres = [...]          # List of genres
languages = [...]       # List of languages
keywords = [...]        # List of corresponding keywords

# Tokenize movie synopses
max_words = 10000  # Consider only the top 10,000 words
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(movie_synopses)
sequences = tokenizer.texts_to_sequences(movie_synopses)
max_sequence_length = max(len(seq) for seq in sequences)

# Pad sequences to have uniform length
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert other features to numpy arrays
release_dates = np.array(release_dates)
genres = np.array(genres)
languages = np.array(languages)

# Combine all features into one input array
input_data = np.column_stack((X, release_dates, genres, languages))

# Convert keywords to one-hot encoding
num_keywords = len(set(keywords))
keyword_mapping = {keyword: i for i, keyword in enumerate(set(keywords))}
y = np.zeros((len(keywords), num_keywords))
for i, keyword in enumerate(keywords):
    y[i, keyword_mapping[keyword]] = 1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, y, test_size=0.2, random_state=42)

# Define RNN model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_sequence_length))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(num_keywords, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

model = SentenceTransformer("all-MiniLM-L6-v2")

