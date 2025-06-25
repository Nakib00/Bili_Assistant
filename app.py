import os, sys, time, webbrowser, requests, pygame, pickle, random, numpy as np, nltk, re, json
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import gensim.downloader as api

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Set the initial language for recognition and response
language = 'en-US'  
use_gtts = False

# Flag to manage whether the system is speaking
speaking = False

# Load intents from both JSON files with utf-8 encoding
intents = []
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents += json.load(json_data)['intents']  # Use += to add list elements

with open('intents_bengali.json', 'r', encoding='utf-8') as json_data:
    intents += json.load(json_data)['intents']  # Use += to add list elements

# Load pre-trained word embeddings
print("Loading word embeddings...")
word_vectors = api.load("glove-wiki-gigaword-100")
print("Word embeddings loaded.")

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Prepare corpus for TF-IDF
corpus = []
for intent in intents:
    if isinstance(intent, dict):
        corpus.extend(intent['patterns'])

# Fit TF-IDF vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)


# Set the initial language for recognition and response
language = 'en-US'  # Default to English
use_gtts = False  # Default to pyttsx3 for English responses

# Flag to manage whether the system is speaking
speaking = False

# Global variables to store the "You said" and "Bot" responses
user_input_data = ""
bot_reply_data = ""
system_status = "Ready"  # Initialize system status

def listen():
    """Capture and recognize speech, returning the recognized text."""
    global language, use_gtts, speaking, user_input_data
    
    # Wait until speaking is finished before listening
    while speaking:
        time.sleep(0.1)
    
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            text = recognizer.recognize_google(audio, language=language)
            print(f"You said: {text}")
            user_input_data = text  # Store user input for Flask display

            # Language switching logic
            if 'bangla' in text.lower() or 'Bangla' in text:
                language = 'bn-BD'
                use_gtts = True
                reply("Switched to Bengali language for recognition and response")
            elif 'english' in text.lower() or 'ইংলিশ' in text:
                language = 'en-US'
                use_gtts = False
                reply("Switched to English language for recognition and response")

            return text
        except sr.WaitTimeoutError:
            print("Listening timed out. Please try again.")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Error from Google Speech Recognition service: {e}")
    
    return None


def reply(response):
    """Use Google Text-to-Speech to reply with the first 10 lines of the given response."""
    global speaking, bot_reply_data
    print(f"Bot: {response}")  # Display full response
    bot_reply_data = response  # Store bot reply for Flask display

    # Limit the response to 10 lines
    limited_response = "\n".join(response.splitlines()[:10])

    # Remove or replace unwanted symbols using regex
    sanitized_response = re.sub(r'[*\-\\\/]', '', limited_response)

    speaking = True
    temp_file = "response.mp3"

    try:
        # Set the language for gTTS; use 'bn' for Bangla and 'en' for English
        lang_code = 'bn' if language == 'bn-BD' else 'en'

        # Create the gTTS object for the selected language
        tts = gTTS(text=sanitized_response, lang=lang_code, slow=False)
        tts.save(temp_file)

        # Initialize pygame mixer and play audio
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():  # Wait for playback to finish
            time.sleep(0.1)

    except Exception as e:
        print(f"Error in text-to-speech: {e}")
    finally:
        # Cleanup
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"Error during cleanup: {e}")
        speaking = False

# Example function to play the audio (you can modify this part)

def play_audio(filename):
    # Add your audio playing logic here
    print(f"Playing {filename}...")

# Set the console to use UTF-8 encoding for Unicode characters
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

# Check for available device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
FILE = "data.pth"
data = torch.load(FILE, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "bili"
current_context = None  # Initialize current_context variable

def preprocess_sentence(sentence):
    """Tokenizes, stems, and lemmatizes the input sentence."""
    tokens = tokenize(sentence)
    stemmed_words = [stemmer.stem(word.lower()) for word in tokens]
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return set(stemmed_words), set(lemmatized_words)

def get_sentence_vector(sentence):
    """Get the average word vector for a sentence."""
    words = sentence.lower().split()
    word_vectors_list = [word_vectors[word] for word in words if word in word_vectors]
    if not word_vectors_list:
        return None
    return np.mean(word_vectors_list, axis=0)

def match_intent(user_input):
    """Match user input to the best intent using TF-IDF, cosine similarity, and word embeddings."""
    stemmed_input, lemmatized_input = preprocess_sentence(user_input)
    input_vector = tfidf_vectorizer.transform([user_input])
    input_embedding = get_sentence_vector(user_input)

    best_match_tag = None
    best_match_score = 0

    for intent in intents:
        if isinstance(intent, dict):
            tag = intent['tag']
            for pattern in intent['patterns']:
                pattern_vector = tfidf_vectorizer.transform([pattern])
                cosine_score = cosine_similarity(input_vector, pattern_vector)[0][0]

                stemmed_pattern, lemmatized_pattern = preprocess_sentence(pattern)
                fuzzy_score = max(
                    fuzz.ratio(user_input.lower(), pattern.lower()),
                    fuzz.partial_ratio(user_input.lower(), pattern.lower())
                ) / 100

                pattern_embedding = get_sentence_vector(pattern)
                embedding_score = 0
                if input_embedding is not None and pattern_embedding is not None:
                    embedding_score = cosine_similarity([input_embedding], [pattern_embedding])[0][0]

                score = (cosine_score + fuzzy_score + embedding_score) / 3

                if score > best_match_score:
                    best_match_score = score
                    best_match_tag = tag

    return best_match_tag, best_match_score

def get_model_response(user_input):
    """Get a response from the custom intent model."""
    
    # Get the predicted intent and probability
    tag, prob = match_intent(user_input)

    # Confidence threshold
    if prob > 0.6:
        for intent in intents:
            if intent['tag'] == tag:
                # Pick a random response from the intent's responses
                model_response = random.choice(intent['responses'])
                
                # Check if the model's response needs to be overridden by a special case
                special_response = handle_special_cases(user_input)
                
                # If there is a special case response, return that; otherwise, return the model's response
                return special_response if special_response else model_response

    # If no valid intent is found, or the confidence is too low, use fallback responses
    fallback_responses = [
        "I'm sorry, I didn't quite understand that. Could you please rephrase?",
        "I'm not sure I follow. Can you explain that in a different way?",
        "I'm having trouble understanding. Could you provide more context?",
        "I apologize, but I'm not familiar with that. Can you try asking something else?",
        "I'm still learning and that's a bit unclear to me. Can you try rephrasing your question?"
    ]
    
    # Return a random fallback response if no valid intent or special case is found
    return random.choice(fallback_responses)


def handle_special_cases(user_input):
    """Handle special cases like opening a specific location map or responding to a general map request."""

    # Mapping of keywords to dropdown values
    location_mapping = {
    'auditorium': 'auditorium',
    'multipurpose': 'multipurposeHall',
    'information': 'informationDesk',
    'lobby': 'lobby',
    'admission': 'admissionOffice',
    'canteen': 'helloCenter',
    'dosa': 'dosaOffice',
    'souvenir': 'souvenirShop',
    'jolil': 'jolilShop',
    'proctor': 'proctorOffice',
    'washroom': 'washroom',
    'food': 'foodCourt',
    'swimming': 'swimmingPool',
    'dmk': 'dmkBuilding',
    'jubilee': 'jubileeBuilding',
    'security': 'securityBox',
    'health': 'HealthCenter',

    # Bangla translations
    'অডিটোরিয়াম': 'auditorium',
    'মাল্টিপারপাস হল': 'multipurposeHall',
    'তথ্য ডেস্ক': 'informationDesk',
    'লবি': 'lobby',
    'ভর্তি অফিস': 'admissionOffice',
    'ক্যান্টিন': 'helloCenter',
    'ডসা অফিস': 'dosaOffice',
    'স্মারক দোকান': 'souvenirShop',
    'জলিল দোকান': 'jolilShop',
    'প্রক্টর অফিস': 'proctorOffice',
    'প্রসাধন কক্ষ': 'washroom',
    'খাবার আদালত': 'foodCourt',
    'সুইমিং পুল': 'swimmingPool',
    'ডিএমকে বিল্ডিং': 'dmkBuilding',
    'জুবিলী বিল্ডিং': 'jubileeBuilding',
    'নিরাপত্তা বাক্স': 'securityBox',
    'স্বাস্থ্য কেন্দ্র': 'HealthCenter'
}


    # Normalize user input for comparison
    normalized_input = user_input.lower()

    # Check for general map request (e.g., "map" or "ম্যাপ")
    if 'map' in normalized_input or 'ম্যাপ' in normalized_input:
        return "Opening the map page."

    # Check for specific location requests
    for keyword, dropdown_value in location_mapping.items():
        if keyword in normalized_input or dropdown_value.replace(" ", "") in normalized_input:
            # Redirect to the Indoor Navigation website with the query parameter
            webbrowser.open(f"http://127.0.0.1:5000/map?destination={dropdown_value}")
            return f"Redirecting to the map for {keyword.capitalize()}."

    return None

# Flask route to show the current user input and bot reply
@app.route("/")
def index():
    return render_template('index.html', user_input=user_input_data, bot_reply=bot_reply_data)

@app.route("/map")
def map():
    return render_template('map.html')

@app.route("/get_data")
def get_data():
    """Endpoint to get the latest user input and bot response."""
    return jsonify({
        'user_input': user_input_data,
        'bot_reply': bot_reply_data
    })

@app.route("/get_system_status", methods=["GET"])
def get_system_status():
    global system_status
    return jsonify({"status": system_status})

@app.route("/set_system_status/<status>", methods=["POST"])
def set_system_status(status):
    global system_status
    system_status = status
    return jsonify({"status": "success"})

if __name__ == "__main__":
    # Initialize pygame (important to do this before using mixer)
    pygame.init()

    # Run the Flask app in a separate thread
    import threading
    flask_thread = threading.Thread(target=lambda: app.run(debug=True, use_reloader=False))
    flask_thread.start()

    while True:
        system_status = "Listening"
        requests.post("http://127.0.0.1:5000/set_system_status/Listening")
        
        user_input = listen()  # Listen to the user
        if user_input:
            system_status = "Processing"
            requests.post("http://127.0.0.1:5000/set_system_status/Processing")
            model_reply = get_model_response(user_input)  # Get response from the local model
            if model_reply:
                system_status = "Speaking"
                requests.post("http://127.0.0.1:5000/set_system_status/Speaking")
                # Check if the response indicates to open the map page
                if "Opening the map page." in model_reply:
                    # Redirect to the map page
                    webbrowser.open("http://127.0.0.1:5000/map")
                else:
                    reply(model_reply)  # Reply using the appropriate voice engine
                    requests.post("http://127.0.0.1:5000/set_system_status/Giving the output")
            else:
                print("No response from the model.")
        
        # Stop listening
        system_status = "Ready"
        requests.post("http://127.0.0.1:5000/set_system_status/Ready")
