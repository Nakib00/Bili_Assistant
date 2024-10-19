# Standard library imports
import os, sys, time, webbrowser, requests, pygame,pickle, random, numpy as np, nltk, re, json

from dotenv import load_dotenv
from flask import Flask, render_template, jsonify
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# Third-party library imports
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


# Initialize Flask app
app = Flask(__name__)

# Initialize the speech recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Set the initial language for recognition and response
language = 'en-US'  # Default to English
use_gtts = False  # Default to pyttsx3 for English responses

# Flag to manage whether the system is speaking
speaking = False

def listen():
    """Capture and recognize speech, returning the recognized text."""
    global language, use_gtts, speaking, user_input_data
    while speaking:  # Wait until speaking is finished before listening
        time.sleep(0.1)  # Slight delay to prevent high CPU usage
    
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise based on the environment
        audio = recognizer.listen(source, phrase_time_limit=15)  # No timeout, just a phrase time limit of 15 seconds
        try:
            text = recognizer.recognize_google(audio, language=language)
            print(f"You said: {text}")
            user_input_data = text  # Store user input for Flask display

            # Switch to Bengali if the word "Bangla" is spoken
            if 'bangla' in text.lower() or 'Bangla' in text:
                language = 'bn-BD'
                use_gtts = True
                print("Switched to Bengali language for recognition and response")

            # Switch back to English if the word "English" or "ইংলিশ" is spoken
            elif 'english' in text.lower() or 'ইংলিশ' in text:
                language = 'en-US'
                use_gtts = False
                print("Switched to English language for recognition and response")

            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
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

    # Set the language for gTTS; use 'bn' for Bangla and 'en' for English
    lang_code = 'bn' if language == 'bn-BD' else 'bn'  # Adjust according to your language variable

    # Create the gTTS object for the selected language
    tts = gTTS(text=sanitized_response, lang=lang_code, slow=False)  # Set slow=False for normal speed
    tts.save("response.mp3")

    # Initialize pygame mixer and play audio
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():  # Wait for playback to finish
        time.sleep(0.1)

    pygame.mixer.music.stop()  # Ensure the audio file is no longer in use
    pygame.mixer.quit()  # Close the mixer
    os.remove("response.mp3")  # Now it is safe to delete the file

    speaking = False

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

    return None  # No special case detected


# Global variables to store the "You said" and "Bot" responses
user_input_data = ""
bot_reply_data = ""

# Set the console to use UTF-8 encoding for Unicode characters
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

# Check for available device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from both JSON files with utf-8 encoding
intents = []
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents += json.load(json_data)['intents']  # Use += to add list elements

with open('intents_bengali.json', 'r', encoding='utf-8') as json_data:
    intents += json.load(json_data)['intents']  # Use += to add list elements

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

# Get  response
def get_model_response(user_input):
    """Get a response from the custom intent model only."""
    
    global current_context  # Use the global variable

    # Check for context and update it
    if current_context and not any(context in user_input for context in current_context):
        user_input = f"{current_context[0]} {user_input}"

    sentence = tokenize(user_input)

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Check if the confidence is high and handle context
    if prob.item() > 0.75:
        response = None  # Initialize response variable

        # Check the type of intents variable to debug
        print(f"Type of intents: {type(intents)}")  # Debugging: check the type

        for intent in intents:  # Iterate through intents
            if isinstance(intent, dict) and tag == intent["tag"]:  # Ensure intent is a dictionary
                response = random.choice(intent['responses'])

                # Check if the intent has a context set
                if "context_set" in intent:
                    current_context = intent["context_set"]
                    break  # Break the loop after finding a matching intent

        return response

    # Handle special cases like map or video
    special_response = handle_special_cases(user_input)
    if special_response:
        return special_response

    # If no special cases or intent matches, return a fallback response
    return "I'm sorry, I didn't understand that. Can you please rephrase?"


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

# Main loop: Update to use the new model-based response
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
