import pandas as pd

# Load movies data
data = pd.read_csv('cleaned_file.csv', low_memory=False)


def extract_keywords(keywords_list):
    """ Extracts a list of keywords from the keywords dictionary. """
    return ' '.join([k['name'] for k in keywords_list])

data['keywords'] = data['keywords'].apply(extract_keywords)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['keywords'])
sequences = tokenizer.texts_to_sequences(data['keywords'])

# Pad sequences to a uniform length
max_seq_length = max(len(x) for x in sequences)
keywords_seq = pad_sequences(sequences, maxlen=max_seq_length)

title_tokenizer = Tokenizer()
title_tokenizer.fit_on_texts(data['title'])
title_seq = title_tokenizer.texts_to_sequences(data['title'])
data['title_encoded'] = [seq[0] if seq else 0 for seq in title_seq]  # Single-word titles
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size for keywords
title_vocab_size = len(title_tokenizer.word_index) + 1  # Vocabulary size for titles

model = Sequential([
    Embedding(vocab_size, 100, input_length=max_seq_length),
    LSTM(50),
    Dense(title_vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(keywords_seq, data['title_encoded'], test_size=0.2)

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
def predict_title(keywords):
    seq = tokenizer.texts_to_sequences([keywords])
    padded = pad_sequences(seq, maxlen=max_seq_length)
    pred = model.predict(padded)
    pred_title_index = pred.argmax()
    return title_tokenizer.index_word[pred_title_index]

# Test the model
user = input("Enter your movie preferences: ")
print(predict_title(user))
