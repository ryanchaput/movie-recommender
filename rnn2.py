import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the merged CSV file into a DataFrame
df = pd.read_csv('merged_file.csv')

# Preprocess the data
# Combine synopsis, keywords, and genre into a single text column
df['text'] = df['overview'].astype(str) + ' ' + df['keywords'].astype(str) + ' ' + df['genres'].astype(str)

# Encode the movie titles as numerical labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['title'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

print(df['text'])

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to a fixed length
max_length = 200
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Build the RNN model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[early_stop])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Make predictions
text_input = "A thrilling science fiction movie set in a dystopian future"
text_sequence = tokenizer.texts_to_sequences([text_input])
padded_sequence = pad_sequences(text_sequence, maxlen=max_length)
predicted_label = model.predict(padded_sequence)
predicted_movie = label_encoder.inverse_transform([predicted_label.argmax()])
print('Predicted Movie:', predicted_movie)