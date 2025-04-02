import random
import pandas as pd
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

animals = ["cat", "dog", "elephant", "tiger", "lion", "rabbit", "bear", "wolf", "zebra", "giraffe", "monkey", "horse", "koala", "kangaroo", "cheetah", "panda", "panther", "penguin"]
colors = ["blue", "red", "green", "yellow", "orange", "purple", "brown", "black", "white", "pink", "gray", "beige", "turquoise", "gold"]
actions = ["runs", "sits", "jumps", "plays", "eats", "sleeps", "climbs", "swims", "chases", "barks", "meows", "roars", "gallops", "hops"]
sizes = ["small", "medium", "large", "tiny", "huge", "gigantic"]
emotions = ["happy", "angry", "excited", "calm", "nervous", "curious", "sad", "playful", "surprised", "fearful"]
locations = ["in the park", "on the road", "under the tree", "by the lake", "in the house", "in the forest", "on the beach", "in the desert", "on the mountain", "at the zoo"]
times = ["in the morning", "at night", "during the day", "at dusk", "at dawn", "at noon", "in the evening"]

def generate_sentence(animal, color, action, size, emotion, location, time):
    prompt = f"A {size} {animal} that is {emotion} and {action}, and is {color}, located {location} {time}."
    result = generator(prompt, max_length=50, num_return_sequences=1)
    generated_text = result[0]['generated_text']
    return generated_text

data = []

for i in range(100): 
    animal_1, color_1, action_1, size_1, emotion_1, location_1, time_1 = (
        random.choice(animals),
        random.choice(colors),
        random.choice(actions),
        random.choice(sizes),
        random.choice(emotions),
        random.choice(locations),
        random.choice(times),
    ) 
    similarity = random.choice([0, 1])

    if similarity == 1:
        context_choice = random.choice(["color", "animal", "action", "location"])
        if context_choice == "color":
            animal_2, action_2, location_2 = random.choice(animals), random.choice(actions), random.choice(locations)
            color_2 = color_1
        elif context_choice == "animal":
            color_2, action_2, location_2 = random.choice(colors), random.choice(actions), random.choice(locations)
            animal_2 = animal_1
        elif context_choice == "action":
            animal_2, color_2, location_2 = random.choice(animals), random.choice(colors), random.choice(locations)
            action_2 = action_1
        else:
            animal_2, color_2, action_2 = random.choice(animals), random.choice(colors), random.choice(actions)
            location_2 = location_1
    else: 
        animal_2, color_2, action_2, size_2, emotion_2, location_2, time_2 = (
            random.choice([a for a in animals if a != animal_1]),
            random.choice([c for c in colors if c != color_1]),
            random.choice([a for a in actions if a != action_1]),
            random.choice([s for s in sizes if s != size_1]),
            random.choice([e for e in emotions if e != emotion_1]),
            random.choice([l for l in locations if l != location_1]),
            random.choice([t for t in times if t != time_1]),
        ) 
    sentence_1 = generate_sentence(animal_1, color_1, action_1, size_1, emotion_1, location_1, time_1)
    sentence_2 = generate_sentence(animal_2, color_2, action_2, size_2, emotion_2, location_2, time_2) 
    data.append({
        "Sentence_1": sentence_1,
        "Sentence_2": sentence_2,
        "Context": context_choice if similarity == 1 else "None",
        "Similarity": similarity
    }) 
df = pd.DataFrame(data)
df.to_csv('dataset2.csv', index=False) 
print(df.head())
