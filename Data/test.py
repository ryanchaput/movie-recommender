import spacy
from spacy.training import Example

nlp = spacy.blank("en")

ner = nlp.add_pipe("ner")

ner.add_label("ARTIST")
ner.add_label("SONG")
ner.add_label("GENRE")
ner.add_label("VIBE") 
ner.add_label("ERA") 
ner.add_label("TEMPO") 
ner.add_label("INSTRUMENT") 
ner.add_label("VOLUME") 

TRAINING_DATA = [
    Example.from_dict(nlp.make_doc("Play a song by The Beatles"), {"entities": [(15, 26, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Find a song I can dance to"), {"entities": [(18, 23, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Play a song from the 70s"), {"entities": [(21, 24, "ERA")]})
    # Rest of your training data...
]

# Train the model
nlp.begin_training()
for epoch in range(10):
    for example in TRAINING_DATA:
        nlp.update([example], losses={})

# Save the trained model
nlp.to_disk("./model")