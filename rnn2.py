import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Load the merged CSV file into a DataFrame
df = pd.read_csv('merged_file.csv')

# Preprocess the data
df['text'] = df['overview'].astype(str) + ' ' + df['genres'].astype(str) + ' ' + df['keywords'].astype(str)

# Encode the movie titles as numerical labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['title'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

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
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.1, mode="min", baseline=None)
model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test), epochs=15, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Make predictions
text_input = "A thrilling science fiction movie set in a dystopian future"
input_embedding = sbert_model.encode([text_input])
input_reshaped = input_embedding.reshape(1, 1, input_embedding.shape[1])
predicted_label = model.predict(input_reshaped)
predicted_movie = label_encoder.inverse_transform([predicted_label.argmax()])
print('Predicted Movie:', predicted_movie)
print('Predicted Movie Overview:', df[df['title'] == predicted_movie[0]]['overview'].values[0])