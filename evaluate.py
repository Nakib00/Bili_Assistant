import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the trained model
model = load_model('model_combined.h5')

# Load intents file to get the classes
with open('intents.json') as file:
    data = json.load(file)

# Extracting the intents for labels (output classes)
classes = [intent['tag'] for intent in data['intents']]

# Load your test data (replace with actual loading of test data)
# Assuming X_test contains your test features, and y_test contains your true labels
X_test = np.load('X_test.npy')  # Replace with actual test feature loading
y_test = np.load('y_test.npy')  # Replace with actual test labels loading

# Predict using the trained model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert the true labels to numerical form using LabelEncoder (used in training)
label_encoder = LabelEncoder()
label_encoder.fit(classes)  # Fit with the same classes used in training

# Convert the test labels to numerical form
y_test_encoded = label_encoder.transform(y_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test_encoded, y_pred_classes)

print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')
