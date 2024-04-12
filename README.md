# movie-recommender

### Dataset
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=keywords.csv

Dataset is a list of movies contianing information like the budget, title, language(s), popularity value, release date, runtime, and more.

### The models
We constructed three recommendation models: a Bayesian model, a neural network model built on our training, and a model powered by SentenceTransform, an existing and pre-trained transformer.
The Bayesian model relied on preprocessed data, vectorizing this data to associate keywords with a genre by noting their frequency. It then takes in the user's input and vectorizes this as well. The model then predicts the probability of each genre/tag based on this processed input and returns the top five most similar movies based on the cosine similarity of the input and movie's vectors.
Our custom trained neural network model was built using the PyTorch library. This model was trained using 80% of our dataset as test