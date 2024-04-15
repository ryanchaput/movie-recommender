# movie-recommender

### Dataset
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=keywords.csv

Dataset is a list of movies contianing information like the budget, title, language(s), popularity value, release date, runtime, and more.

### The models
1. RNN built by us using TensorFlow and Keras frameworks. Trained model using the synopsis (overview), keywords, and genre of the film, using 80% of the dataset for training.
2. Bayesian model using SentenceTransform (SentenceBERT) to create embeddings of movie synopsis from dataset. Model then takes user input and creates an ebedding from it.
3. Transform with cosine similarity: uses SentenceTransform (SentenceBERT) to create embeddings from dataset. Model then creates an embedding from user input and finds the five most similar movies based on the cosine similarities of these embeddings. This model is the most accurate one we created.