# Bili AI Chatbot with Indoor Navigation

This is a chatbot application with speech recognition, response synthesis, and indoor navigation functionality. It supports both English and Bengali languages for voice recognition and responses, and can direct users to specific locations via an indoor map.

## Features

- **Speech Recognition**: Captures user input through voice using Google Speech Recognition (supports English and Bengali).
- **Text-to-Speech Responses**: Provides voice responses using either `pyttsx3` for English or `gTTS` for Bengali.
- **Intent Matching**: Matches user input to predefined intents using TF-IDF, cosine similarity, fuzzy matching, and word embeddings (`GloVe`).
- **Indoor Navigation**: Opens a web-based indoor map with route guidance to various locations.
- **Language Switching**: Users can switch between English and Bengali during conversation.

## Requirements

To run the project, ensure you have the following dependencies installed:

- Python 3.x
- Flask
- Pygame
- Google Text-to-Speech (gTTS)
- SpeechRecognition
- PyTorch
- FuzzyWuzzy
- Gensim (GloVe word embeddings)
- Scikit-learn
- NLTK
- dotenv


### start install
To create a `.venv` (virtual environment) and install dependencies from a `requirements.txt` file using Git Bash, follow these steps:

### 1. **Create a virtual environment named `.venv`**:
   Run the following command in Git Bash:
   ```bash
   python -m venv .venv
   ```

### 2. **Activate the virtual environment**:
   - On **Windows** (using Git Bash):
     ```bash
     source .venv/Scripts/activate
     ```
   - On **Linux/MacOS** (if using Git Bash in a Unix-based environment):
     ```bash
     source .venv/bin/activate
     ```

### 3. **Install dependencies from `requirements.txt`**:
   After activating the virtual environment, install the packages from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

Now, your virtual environment will be set up and all the required packages will be installed!


### app.py

This is the main application file responsible for initializing the Flask web server and handling user interactions.

![Home Interface](https://github.com/Nakib00/Bili_Assistant/blob/main/assets/home.png?raw=true)

![Home Interface](https://github.com/Nakib00/Bili_Assistant/blob/main/assets/Processing.png?raw=true)

![Home Interface](https://github.com/Nakib00/Bili_Assistant/blob/main/assets/reply.png?raw=true)

![Map Interface](https://github.com/Nakib00/Bili_Assistant/blob/main/assets/map.png?raw=true)


## Files and Their Purpose

### 1. `train.py`
This script is responsible for training the chatbot model. It performs the following tasks:

- **Loads and processes intent data** from multiple JSON files.
- **Prepares training data** by tokenizing, stemming, and converting the data into a bag-of-words format.
- **Defines and trains** a neural network model for intent classification.
- **Saves the trained model** and metadata required for inference.

### 2. `nltk_utils.py`
This file provides utility functions for text preprocessing. These include:
- Tokenizing input sentences.
- Stemming words to their root form.
- Generating a bag-of-words representation of the input.

### 3. `model.py`
This file defines the neural network architecture used for intent classification. It contains a simple feedforward neural network with two hidden layers and ReLU activation.

---

## `train.py`

### Data Loading
The script loads intent data from two JSON files: `intents.json` and `intents_bengali.json`. These files contain various user intents, each associated with patterns (example user inputs) and responses. The `json` module is used to parse the files, and the intents are stored in a list for further processing.

### Data Preprocessing
- **Tokenizing and Stemming**: The script tokenizes each pattern (user input) into words and stems them to get their root form using the functions from `nltk_utils.py`.
- **Bag of Words**: Each pattern is converted into a bag-of-words representation, which is used as input to the model.
- **Label Encoding**: The target labels (intents/tags) are stored in the `tags` list and encoded as numerical indices for training.

### Model Training
The script defines a neural network model (`NeuralNet` from `model.py`) and trains it using the cross-entropy loss function and the Adam optimizer. The model is trained for a specified number of epochs.

### Saving the Model
After training, the model’s state and the necessary metadata (e.g., vocabulary and tag list) are saved in a file (`data.pth`) for later use during inference.

---

## `nltk_utils.py`

### Tokenization
The `tokenize` function splits a sentence into words/tokens, which can include words, punctuation marks, or numbers.

### Stemming
The `stem` function reduces words to their root form using the `PorterStemmer` from the `nltk` library.

### Bag of Words
The `bag_of_words` function converts a tokenized sentence into a bag-of-words vector. For each word in the vocabulary, it assigns `1` if the word is present in the sentence and `0` otherwise.

---

## `model.py`

### Neural Network Architecture
The `NeuralNet` class defines a simple feedforward neural network, which includes:

- **Input Layer**: Takes the bag-of-words vector as input.
- **Two Hidden Layers**: Uses ReLU as the activation function.
- **Output Layer**: Outputs raw scores (logits) for each intent class, without applying softmax.

---

