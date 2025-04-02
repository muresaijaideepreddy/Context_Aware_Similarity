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
    return result[0]['generated_text']

data = []

# Generate similar sentence pairs (label = 1)
for i in range(5): 
    animal_1, color_1, action_1, size_1, emotion_1, location_1, time_1 = random.choice(animals), random.choice(colors), random.choice(actions), random.choice(sizes), random.choice(emotions), random.choice(locations), random.choice(times)
    animal_2, color_2, action_2, size_2, emotion_2, location_2, time_2 = animal_1, color_1, action_1, size_1, emotion_1, location_1, time_1  # Ensure similarity

    # Identify matching contexts
    matching_contexts = []
    if animal_1 == animal_2:
        matching_contexts.append("animal")
    if color_1 == color_2:
        matching_contexts.append("color")
    if action_1 == action_2:
        matching_contexts.append("action")
    if size_1 == size_2:
        matching_contexts.append("size")
    if emotion_1 == emotion_2:
        matching_contexts.append("emotion")
    if location_1 == location_2:
        matching_contexts.append("location")
    if time_1 == time_2:
        matching_contexts.append("time")

    # Store each matching context
    for context in matching_contexts:
        sentence_1 = generate_sentence(animal_1, color_1, action_1, size_1, emotion_1, location_1, time_1)
        sentence_2 = generate_sentence(animal_2, color_2, action_2, size_2, emotion_2, location_2, time_2)
        data.append([sentence_1, sentence_2, context, 1])

# Generate dissimilar sentence pairs (label = 0)
for i in range(5): 
    animal_1, color_1, action_1, size_1, emotion_1, location_1, time_1 = random.choice(animals), random.choice(colors), random.choice(actions), random.choice(sizes), random.choice(emotions), random.choice(locations), random.choice(times)
    
    # Ensure at least one attribute is different
    animal_2, color_2, action_2, size_2, emotion_2, location_2, time_2 = random.choice(animals), random.choice(colors), random.choice(actions), random.choice(sizes), random.choice(emotions), random.choice(locations), random.choice(times)
    
    # Find matching contexts and ensure differences
    while (animal_2 == animal_1 and color_2 == color_1 and action_2 == action_1 and size_2 == size_1 and emotion_2 == emotion_1 and location_2 == location_1 and time_2 == time_1):
        animal_2, color_2, action_2, size_2, emotion_2, location_2, time_2 = random.choice(animals), random.choice(colors), random.choice(actions), random.choice(sizes), random.choice(emotions), random.choice(locations), random.choice(times)

    # Identify differing contexts
    differing_contexts = []
    if animal_1 != animal_2:
        differing_contexts.append("animal")
    if color_1 != color_2:
        differing_contexts.append("color")
    if action_1 != action_2:
        differing_contexts.append("action")
    if size_1 != size_2:
        differing_contexts.append("size")
    if emotion_1 != emotion_2:
        differing_contexts.append("emotion")
    if location_1 != location_2:
        differing_contexts.append("location")
    if time_1 != time_2:
        differing_contexts.append("time")

    # Store each differing context
    for context in differing_contexts:
        sentence_1 = generate_sentence(animal_1, color_1, action_1, size_1, emotion_1, location_1, time_1)
        sentence_2 = generate_sentence(animal_2, color_2, action_2, size_2, emotion_2, location_2, time_2)
        data.append([sentence_1, sentence_2, context, 0])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Sentence 1", "Sentence 2", "Context", "Similarity"])
print(df.head())  # Display first few rows
df.to_csv('dataset2.csv', index=False)  # Save to CSV
