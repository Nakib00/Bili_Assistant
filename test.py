import json
import numpy as np
import pickle
from keras.models import load_model
import nltk
from nltk.tokenize import word_tokenize
import random

# Load the trained combined model
model = load_model("model_combined.h5")

# Load words and classes
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Load intents JSON files for both languages
with open("intents.json") as file_en:
    intents_en = json.load(file_en)

with open("intents_bengali.json", encoding="utf-8") as file_bn:
    intents_bn = json.load(file_bn)

# Combine intents from both languages
intents = {'intents': intents_en['intents'] + intents_bn['intents']}

# Function to clean up the input sentence and convert it into a bag of words
def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence.lower())  # Tokenize the sentence
    bag = [0] * len(words)  # Create a bag of words of the same length as the vocabulary

    for word in sentence_words:
        if word in words:
            bag[words.index(word)] = 1  # Mark 1 if the word exists in the vocabulary
    return np.array(bag)

# Function to predict the intent of the input sentence
def predict_class(sentence):
    bag_of_words = clean_up_sentence(sentence)
    res = model.predict(np.array([bag_of_words]))[0]  # Predict the class

    # Get the index of the highest probability class
    threshold = 0.25  # Only consider predictions above a certain probability threshold
    results = [[i, r] for i, r in enumerate(res) if r > threshold]

    if results:
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by highest probability
        predicted_classes = [(classes[i], prob) for i, prob in results]
        return predicted_classes  # Return list of predicted classes with their probabilities
    else:
        return None  # Return None if no intent is predicted with sufficient confidence

# Function to get a response based on the predicted intent (tag)
def get_response(predicted_tag):
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])  # Return a random response from the list of responses
    return "Sorry, I didn't quite understand that. Can you please rephrase?"

# Enhanced error handling and better responses
def handle_response(predicted_classes):
    if predicted_classes:
        # Check confidence of the top class
        top_class, top_confidence = predicted_classes[0]
        
        # If the top intent confidence is strong enough, return the corresponding response
        if top_confidence > 0.75:
            return get_response(top_class)
        elif len(predicted_classes) > 1:
            # If the confidence is lower but there are other close matches, suggest clarification
            second_class, second_confidence = predicted_classes[1]
            return f"Did you mean '{top_class}' or '{second_class}'?"
        else:
            return f"I'm not very sure. Did you mean '{top_class}'?"
    else:
        return "Sorry, I didn't understand that. Can you please try again?"

# Test the model with an input sentence
while True:
    user_input = input("Enter a sentence in English or Bengali (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    # Predict the intent(s) with their confidence scores
    predicted_intents = predict_class(user_input)
    
    # Handle response based on predictions
    if predicted_intents:
        response = handle_response(predicted_intents)
        print(f"Bot response: {response}")
    else:
        print("Sorry, I couldn't find an appropriate response.")
