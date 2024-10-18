import random
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Initialize lists to store data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.']

# Function to load and preprocess intents data
def load_intents(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            intents = json.load(file)
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word in the sentence
                word_list = word_tokenize(pattern)
                word_list = [w for w in word_list if w not in stop_words]  # Remove stopwords
                words.extend(word_list)
                # Add the tokenized sentence and the intent tag to documents
                documents.append((word_list, intent['tag']))
                # Add to classes if the tag is not already present
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])
    except Exception as e:
        print(f"Error loading intents from {file_path}: {e}")

# Load intents from both JSON files
load_intents('intents.json')
load_intents('intents_bengali.json')

# Lemmatize and lower each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes alphabetically
classes = sorted(list(set(classes)))

# Print basic information
print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Save words and classes to files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create the training data
training = []
output_empty = [0] * len(classes)

# Create the training set: bag of words for each sentence
for doc in documents:
    bag = []
    pattern_words = doc[0]
    # Lemmatize the words in the current pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create the bag of words array with 1 if word is present in the pattern, otherwise 0
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    
    # Output is a '0' for each tag and '1' for the current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Split the data into X (patterns) and Y (intents)
train_x = np.array([item[0] for item in training], dtype=np.float32)  # Bags of words (input data)
train_y = np.array([item[1] for item in training], dtype=np.float32)  # One-hot encoded intents (labels)

# Print the size of the training data
print(f"Training data created with shapes: train_x: {train_x.shape}, train_y: {train_y.shape}")

# Build the Sequential model
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))  # Increased number of neurons
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model with Adam optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=3000, batch_size=8, verbose=1, validation_split=0.2)

# Save the trained model
model.save('model_combined.h5')

# Save training history to a file for analysis
with open('training_history.json', 'w') as file:
    json.dump(hist.history, file)

print("Model trained and saved as model_combined.h5")
