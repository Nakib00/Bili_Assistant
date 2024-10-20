# Bili-Assistant

Overview
--------

This application is a voice assistant built using Flask and various libraries that facilitate speech recognition, text-to-speech, and machine learning for natural language processing. The assistant can understand user commands, switch between languages (English and Bengali), and respond appropriately, including opening specific maps or providing predefined responses.

### Key Features

*   Speech recognition using Google Speech Recognition.
    
*   Text-to-speech responses using gTTS (Google Text-to-Speech) and pyttsx3.
    
*   Language switching capability between English and Bengali.
    
*   Custom intent classification using a trained TensorFlow model.
    
*   Dynamic response generation based on user input.
    
*   Integration with a web interface using Flask.
    

File Structure


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


#### Dependencies

*   Flask
    
*   TensorFlow
    
*   NLTK
    
*   SpeechRecognition
    
*   pyttsx3
    
*   gTTS
    
*   pygame
    
*   numpy
    
*   requests
    
*   dotenv
    
*   json
    
*   pickle
    
*   random
    
*   re
    
*   os
    
*   sys
    

#### Main Components

1.  **Global Variables**:
    
    *   user\_input\_data: Stores the last user input for display.
        
    *   bot\_reply\_data: Stores the last bot reply for display.
        
    *   language: Tracks the current language for responses.
        
    *   speaking: Boolean flag indicating whether the system is currently speaking.
        
2.  **Functions**:
    
    *   listen(): Captures audio input from the user and recognizes it using Google Speech Recognition. Adjusts language based on detected keywords.
        
    *   reply(response): Responds to the user using text-to-speech based on the model's reply.
        
    *   handle\_special\_cases(user\_input): Handles specific commands related to maps and locations.
        
    *   clean\_up\_sentence(sentence): Tokenizes and lemmatizes input sentences for processing.
        
    *   bow(sentence, words, show\_details=True): Creates a bag-of-words representation of the input sentence.
        
    *   predict\_class(sentence, model): Predicts the intent of the input sentence using the trained model.
        
    *   getResponse(ints, intents\_json, intents\_bangali\_json): Retrieves a response based on the predicted intent.
        
    *   get\_model\_response(user\_input): Main function to get a response from the trained model or handle special cases.
        
3.  **Flask Routes**:
    
    *   /: Renders the main interface, showing user input and bot replies.
        
    *   /map: Renders a map interface.
        
    *   /get\_data: Returns the latest user input and bot reply in JSON format.
        
    *   /get\_system\_status: Returns the current status of the system.
        
4.  **Main Loop**:
    
    *   Initializes the system, listens for user input, processes the input, and provides responses accordingly.
        


### train.py

This file contains the code for training the natural language processing model used in the voice assistant.

#### Dependencies

*   NLTK
    
*   TensorFlow
    
*   numpy
    
*   json
    
*   random
    
*   pickle
    

#### Main Components

1.  **Global Variables**:
    
    *   words: Stores the vocabulary used in the model.
        
    *   classes: Stores the different intent classes.
        
    *   documents: Contains the training data.
        
    *   ignore\_words: Words to ignore during training (e.g., punctuation).
        
2.  **Functions**:
    
    *   load\_intents(file\_path): Loads and preprocesses intents data from a JSON file. This data is crucial for training the model as it defines the intents and their associated patterns and responses.
        
3.  **NLTK Downloads**:
    
    *   The script automatically downloads necessary resources such as tokenizers, lemmatizers, and stopwords from NLTK.
        
4.  **Model Training**:
    
    *   The training logic should follow in this file (not fully included in the snippet), including building and training the neural network model, saving it for later use.Conclusion
        
