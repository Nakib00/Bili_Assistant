import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Check for available device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from both JSON files with utf-8 encoding
intents = []
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents += json.load(json_data)['intents']  # Use += to add list elements

with open('intents_bengali.json', 'r', encoding='utf-8') as json_data:
    intents += json.load(json_data)['intents']  # Use += to add list elements

# Debugging: Print the structure of loaded intents
print(f"Loaded intents: {intents}")  # Add this line to check structure

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

def get_response(msg):
    global current_context  # Use the global variable

    # Check for context and update it
    if current_context and not any(context in msg for context in current_context):
        msg = f"{current_context[0]} {msg}"

    sentence = tokenize(msg)

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

    return "I do not understand..."

def run_chatbot():
    print(f"{bot_name}: Hello! I am here to assist you. Type 'exit' to quit the chat.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print(f"{bot_name}: Goodbye!")
            break

        response = get_response(user_input)
        print(f"{bot_name}: {response}")

if __name__ == "__main__":
    run_chatbot()
