import spacy
from spacy.training import Example

nlp = spacy.blank("en")

ner = nlp.add_pipe("ner")

# Add labels for movie categories
ner.add_label("")

TRAINING_DATA = []

#msg = spacy.training.offsets_to_biluo_tags(nlp.make_doc("Play a song by The Beatles"), [(0, 4, "ARTIST")])
print("Training")

# Train the model
nlp.begin_training()
for epoch in range(10):
    for example in TRAINING_DATA:
        nlp.update([example], losses={})

# Save the trained model
nlp.to_disk("./model")