import spacy
from spacy.training import Example

nlp = spacy.blank("en")

ner = nlp.add_pipe("ner")

ner.add_label("ARTIST")
ner.add_label("SONG")
ner.add_label("GENRE")
ner.add_label("VIBE") # Emotion/feel of a song, ex. happy, upbeat, sad, dance
ner.add_label("ERA") # Time period of a song, ex. 70s
ner.add_label("TEMPO") # Speed of a song, ex. fast, slow
ner.add_label("INSTRUMENT") # Instrument used in a song, ex. guitar, piano
ner.add_label("VOLUME") # Volume of a song, ex. loud, quiet

TRAINING_DATA = [
    Example.from_dict(nlp.make_doc("Play a song by The Beatles"), {"entities": [(15, 26, "ARTIST")]}),
    """Example.from_dict(nlp.make_doc("Find a song I can dance to"), {"entities": [(18, 24, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Play a song from the 70s"), {"entities": [(18, 21, "ERA")]}),
    Example.from_dict(nlp.make_doc("Play a song by Taylor Swift"), {"entities": [(13, 25, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by The Weeknd"), {"entities": [(13, 22, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by The Chainsmokers"), {"entities": [(13, 27, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("I want a rock song from the 80s"), {"entities": [(9, 13, "GENRE"), (28, 31, "ERA")]}),
    Example.from_dict(nlp.make_doc("Play a song by Queen"), {"entities": [(13, 18, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by The Rolling Stones"), {"entities": [(13, 29, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("I'm looking for a pop song from the 90s"), {"entities": [(16, 19, "GENRE"), (30, 33, "ERA")]}),
    Example.from_dict(nlp.make_doc("Play a song by Michael Jackson"), {"entities": [(13, 27, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by Justin Bieber"), {"entities": [(13, 24, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by Ariana Grande"), {"entities": [(13, 25, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by Drake"), {"entities": [(13, 18, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by Post Malone"), {"entities": [(13, 24, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by Khalid"), {"entities": [(13, 19, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by Ed Sheeran"), {"entities": [(13, 23, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by Shawn Mendes"), {"entities": [(13, 24, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by Sam Smith"), {"entities": [(13, 22, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by Billie Eilish from 2021"), {"entities": [(13, 25, "ARTIST"), (29, 33, "ERA")]}),
    Example.from_dict(nlp.make_doc("Play a song by Dua Lipa"), {"entities": [(13, 21, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("Play a song by Harry Styles"), {"entities": [(13, 24, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("I like the song Wild Horses by The Rolling Stones"), {"entities": [(16, 27, "SONG"), (31, 46, "ARTIST")]}),
    Example.from_dict(nlp.make_doc("I want to listen to an acoustic song"), {"entities": [(23, 31, "VIBE")]}),
    Example.from_dict(nlp.make_doc("I want to listen to a happy song"), {"entities": [(23, 28, "VIBE")]}),
    Example.from_dict(nlp.make_doc("I want to listen to a sad song"), {"entities": [(23, 26, "VIBE")]}),
    Example.from_dict(nlp.make_doc("I want to listen to a dance song"), {"entities": [(23, 28, "VIBE")]}),
    Example.from_dict(nlp.make_doc("I want to listen to a song from the 60s"), {"entities": [(23, 26, "ERA")]}),
    Example.from_dict(nlp.make_doc("I want to listen to a song from the 70s"), {"entities": [(23, 26, "ERA")]}),
    Example.from_dict(nlp.make_doc("I want to listen to a song from the 80s"), {"entities": [(23, 26, "ERA")]}),
    Example.from_dict(nlp.make_doc("I want to listen to a song from the 90s"), {"entities": [(23, 26, "ERA")]}),
    Example.from_dict(nlp.make_doc("I want to listen to a song from the 2000s"), {"entities": [(23, 27, "ERA")]}),
    Example.from_dict(nlp.make_doc("I want to listen to a song from the 2010s"), {"entities": [(23, 27, "ERA")]}),
    Example.from_dict(nlp.make_doc("I want to listen to a song from the 2020s"), {"entities": [(23, 27, "ERA")]}),
    Example.from_dict(nlp.make_doc("Recommend something upbeat"), {"entities": [(15, 21, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a club"), {"entities": [(25, 29, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a party"), {"entities": [(25, 30, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a wedding"), {"entities": [(25, 32, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a funeral"), {"entities": [(25, 32, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a live concert"), {"entities": [(25, 37, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a festival"), {"entities": [(25, 33, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a rave"), {"entities": [(25, 29, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a quiet bar"), {"entities": [(25, 30, "VOLUME"), (31, 34, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a lounge"), {"entities": [(25, 31, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a coffee shop"), {"entities": [(25, 36, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a gym"), {"entities": [(25, 28, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a yoga class"), {"entities": [(25, 34, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Recommend something I would hear at a spa"), {"entities": [(25, 28, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Find something fast and loud"), {"entities": [(13, 17, "TEMPO"), (22, 26, "VOLUME")]}),
    Example.from_dict(nlp.make_doc("Find something slow and quiet"), {"entities": [(13, 17, "TEMPO"), (22, 27, "VOLUME")]}),
    Example.from_dict(nlp.make_doc("Find something upbeat and happy"), {"entities": [(13, 19, "VIBE"), (24, 29, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Find something sad and slow"), {"entities": [(13, 16, "VIBE"), (21, 25, "TEMPO")]}),
    Example.from_dict(nlp.make_doc("Find something danceable and upbeat"), {"entities": [(13, 22, "VIBE"), (27, 33, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Find something I can relax to"), {"entities": [(13, 21, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Find something I can study to"), {"entities": [(13, 21, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Find something I can sleep to"), {"entities": [(13, 21, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Find something I can work out to"), {"entities": [(13, 21, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Find something I can meditate to"), {"entities": [(13, 21, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Find something I can drive to"), {"entities": [(13, 21, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Find something fun I can party to"), {"entities": [(13, 16, "VIBE"), (21, 26, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Find something I can sing along to"), {"entities": [(13, 21, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Find something I can cry to"), {"entities": [(13, 21, "VIBE")]}),
    Example.from_dict(nlp.make_doc("Play a Van Halen song with guitar"), {"entities": [(7, 16, "ARTIST"), (27, 33, "INSTRUMENT")]}),
    Example.from_dict(nlp.make_doc("Play a Beethoven song with piano"), {"entities": [(7, 16, "ARTIST"), (27, 32, "INSTRUMENT")]}),
    Example.from_dict(nlp.make_doc("Play a song by The Beatles with drums"), {"entities": [(15, 24, "ARTIST"), (30, 35, "INSTRUMENT")]}),
    Example.from_dict(nlp.make_doc("Play a song by The Chainsmokers with synthesizer"), {"entities": [(15, 27, "ARTIST"), (33, 45, "INSTRUMENT")]}),
    Example.from_dict(nlp.make_doc("Play a song by Taylor Swift with banjo"), {"entities": [(15, 25, "ARTIST"), (31, 36, "INSTRUMENT")]}),
    Example.from_dict(nlp.make_doc("Play a song by The Weeknd with saxophone"), {"entities": [(15, 22, "ARTIST"), (28, 38, "INSTRUMENT")]}),
    Example.from_dict(nlp.make_doc("Play a song by Queen with bass guitar"), {"entities": [(15, 20, "ARTIST"), (30, 41, "INSTRUMENT")]}),
    Example.from_dict(nlp.make_doc("Play a song by The Rolling Stones with harmonica"), {"entities": [(15, 33, "ARTIST"), (39, 48, "INSTRUMENT")]}),
    Example.from_dict(nlp.make_doc("Recommend something with a groovy baseline", {"entities": [(27, 32, "VIBE"), (33, 40, "INSTRUMENT")]})),
    Example.from_dict(nlp.make_doc("Recommend something with a catchy melody", {"entities": [(27, 32, "VIBE"), (33, 38, "INSTRUMENT")]})),
    Example.from_dict(nlp.make_doc("Recommend something with a smooth saxophone solo", {"entities": [(27, 32, "VIBE"), (33, 41, "INSTRUMENT")]})),
    Example.from_dict(nlp.make_doc("Recommend something with a fast tempo", {"entities": [(25, 38, "VIBE"), (25, 29, "TEMPO")]})),
    Example.from_dict(nlp.make_doc("Recommend something with a slow tempo", {"entities": [(25, 38, "VIBE"), (25, 29, "TEMPO")]})),
    Example.from_dict(nlp.make_doc("Recommend something with a loud volume", {"entities": [(25, 38, "VIBE"), (25, 30, "VOLUME")]})),
    Example.from_dict(nlp.make_doc("Recommend something with a quiet volume", {"entities": [(25, 38, "VIBE"), (25, 31, "VOLUME")]})),
    Example.from_dict(nlp.make_doc("Recommend something with a guitar solo", {"entities": [(25, 38, "VIBE"), (25, 35, "INSTRUMENT")]})),
    Example.from_dict(nlp.make_doc("Recommend something with a piano solo", {"entities": [(25, 38, "VIBE"), (25, 34, "INSTRUMENT")]})),
    Example.from_dict(nlp.make_doc("Recommend something with a drum solo", {"entities": [(25, 38, "VIBE"), (25, 33, "INSTRUMENT")]})),
    Example.from_dict(nlp.make_doc("Give me something jazzy", {"entities": [(17, 22, "GENRE"), (17, 22, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Give me something bluesy", {"entities": [(17, 23, "GENRE"), (17, 23, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Give me something funky", {"entities": [(17, 22, "GENRE"), (17, 22, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Give me something soulful", {"entities": [(17, 23, "GENRE"), (17, 23, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Give me something country", {"entities": [(17, 24, "GENRE"), (17, 24, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Give me something classical", {"entities": [(17, 25, "GENRE"), (17, 25, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Give me something electronic", {"entities": [(17, 26, "GENRE"), (17, 26, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Give me something indie", {"entities": [(17, 21, "GENRE"), (17, 21, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Give me something punk by The Ramones", {"entities": [(17, 21, "GENRE"), (17, 21, "VIBE"), (25, 36, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Give me something metal by Metallica", {"entities": [(17, 22, "GENRE"), (17, 22, "VIBE"), (25, 34, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Give me something hip hop by Kanye West", {"entities": [(17, 24, "GENRE"), (17, 24, "VIBE"), (28, 38, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Give me something rap by Eminem", {"entities": [(17, 20, "GENRE"), (17, 20, "VIBE"), (24, 30, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Give me something R&B by Beyonce", {"entities": [(17, 20, "GENRE"), (17, 20, "VIBE"), (24, 31, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Give me something reggae by Bob Marley", {"entities": [(17, 23, "GENRE"), (17, 23, "VIBE"), (27, 37, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Give me something pop by Lady Gaga", {"entities": [(17, 20, "GENRE"), (17, 20, "VIBE"), (24, 33, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Give me something rock by Led Zeppelin", {"entities": [(17, 21, "GENRE"), (17, 21, "VIBE"), (25, 36, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Give me something folk by Bob Dylan", {"entities": [(17, 21, "GENRE"), (17, 21, "VIBE"), (25, 34, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Find an older Kanye West song", {"entities": [(10, 21, "ARTIST"), (10, 15, "ERA")]})),
    Example.from_dict(nlp.make_doc("Find a newer Kanye West song", {"entities": [(10, 21, "ARTIST"), (10, 15, "ERA")]})),
    Example.from_dict(nlp.make_doc("Find an older Taylor Swift song", {"entities": [(10, 21, "ARTIST"), (10, 15, "ERA")]})),
    Example.from_dict(nlp.make_doc("Find a newer Taylor Swift song", {"entities": [(10, 21, "ARTIST"), (10, 15, "ERA")]})),
    Example.from_dict(nlp.make_doc("Find a popular Led Zeppelin song", {"entities": [(7, 14, "VIBE"), (16, 24, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Find a popular Beatles song", {"entities": [(7, 14, "VIBE"), (16, 24, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Find a popular Michael Jackson song", {"entities": [(7, 14, "VIBE"), (16, 29, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Find a popular Queen song", {"entities": [(7, 14, "VIBE"), (16, 20, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Find a less popular Rolling Stones song", {"entities": [(7, 19, "VIBE"), (20, 35, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Find a less popular Taylor Swift song", {"entities": [(7, 19, "VIBE"), (20, 31, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Find a less popular Ariana Grande song", {"entities": [(7, 19, "VIBE"), (20, 25, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Find a less popular Drake song", {"entities": [(7, 19, "VIBE"), (20, 25, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Find a less popular Post Malone song", {"entities": [(7, 19, "VIBE"), (20, 29, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Find a less popular Khalid song", {"entities": [(7, 19, "VIBE"), (20, 26, "ARTIST")]})),
    Example.from_dict(nlp.make_doc("Play something like The Beatles", {"entities": [(18, 27, "ARTIST"), (18, 27, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Play something like The Rolling Stones", {"entities": [(18, 33, "ARTIST"), (18, 33, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Play something like Fearless by Taylor Swift", {"entities": [(18, 25, "SONG"), (29, 40, "ARTIST"), (18, 25, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Play something like Blinding Lights by The Weeknd", {"entities": [(18, 32, "SONG"), (36, 45, "ARTIST"), (18, 32, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Play something like Closer by The Chainsmokers", {"entities": [(18, 24, "SONG"), (28, 41, "ARTIST"), (18, 24, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Play something like Bohemian Rhapsody by Queen", {"entities": [(18, 37, "SONG"), (41, 46, "ARTIST"), (18, 37, "VIBE")]})),
    Example.from_dict(nlp.make_doc("Play something like Wild Horses by The Rolling Stones", {"entities": [(18, 29, "SONG"), (33, 48, "ARTIST"), (18, 29, "VIBE")]})),"""
]

#msg = spacy.training.offsets_to_biluo_tags(nlp.make_doc("Play a song by The Beatles"), [(0, 4, "ARTIST")])
print("Training")

# Train the model
nlp.begin_training()
for epoch in range(10):
    for example in TRAINING_DATA:
        nlp.update([example], losses={})

# Save the trained model
nlp.to_disk("./model")