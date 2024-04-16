# movie-recommender

### Dataset
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=keywords.csv

Dataset is a list of movies contianing information like the budget, title, language(s), popularity value, release date, runtime, and more.

### The models
1. RNN built by us using TensorFlow and Keras frameworks. Trained model using the synopsis (overview), keywords, and genre of the film, using 80% of the dataset for training (about 24,500 data points). This model did not perform as well as we had hoped, which can be blamed on limited training. If we were to continue our work on this model it would require a much larger dataset with more focused training.
2. Bayesian model using SentenceTransform (SentenceBERT) to create embeddings of movie synopsis (overview) from dataset. These embeddings are then used to train a **Gaussian Naive Bayes classifier**, associating the synopsis (overview) with genre(s). The model then takes the user input and creates an embedding from it, treating it like a movie overview. The model then uses this embedding to predict what genre of movie the user is looking for, and recommends one that matches this genre at random.
3. Transform with cosine similarity: uses SentenceTransform (SentenceBERT) to create embeddings from the movie synopsis (overview). The model then creates an embedding from user input, treating it like a general synopsis of a movie the user is looking for, and finds the five most similar movies based on the **cosine similarities** of these embeddings. This model is the most accurate one we created. However, it could still be improved as it treats the user's input the same as a movie synopsis. This could be improved in the future to be treateed like a movie's *description* rather than pure synopsis, but would require better data to be available.