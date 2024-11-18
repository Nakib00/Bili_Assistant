import torch
import numpy as np
import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']

# Load model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(data['model_state'])
model.eval()

# Load intents
intents_files = ['intents.json', 'intents_bengali.json']
intents = []
for file in intents_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        intents.extend(data['intents'])

# Prepare test data
test_sentences = []
test_labels = []

for intent in intents:
    for pattern in intent['patterns']:
        test_sentences.append(pattern)
        test_labels.append(intent['tag'])

# Tokenize and create bag of words for test data
X_test = []
y_test = []
for sentence, label in zip(test_sentences, test_labels):
    bag = bag_of_words(tokenize(sentence), all_words)
    X_test.append(bag)
    y_test.append(tags.index(label))

X_test = np.array(X_test)
y_test = np.array(y_test)

# Evaluation metrics storage
metrics = []

# Define classifiers
classifiers = {
    "NeuralNet": model,
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True),
    "KNeighbors": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB()
}

# Function to evaluate the classifier
def evaluate_classifier(clf, X, y, classifier_name):
    if classifier_name == "NeuralNet":
        # NeuralNet predictions
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            outputs = clf(X_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.numpy()
    else:
        # Fit and predict for sklearn models
        clf.fit(X, y)
        predicted = clf.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y, predicted)
    precision = precision_score(y, predicted, average='weighted')
    recall = recall_score(y, predicted, average='weighted')
    f1 = f1_score(y, predicted, average='weighted')

    # Append results with two decimal precision
    metrics.append({
        'Model': classifier_name,
        'Accuracy': round(accuracy, 2),
        'Precision': round(precision, 2),
        'Recall': round(recall, 2),
        'F1 Score': round(f1, 2)
    })

# Evaluate each classifier
for name, clf in classifiers.items():
    evaluate_classifier(clf, X_test, y_test, name)

# Convert metrics to a DataFrame and save as Excel file
metrics_df = pd.DataFrame(metrics)
output_dir = "evaluation_metrics"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

excel_path = os.path.join(output_dir, 'evaluation_metrics.xlsx')
with pd.ExcelWriter(excel_path) as writer:
    metrics_df.to_excel(writer, index=False, sheet_name='Metrics')

print("Evaluation complete. Metrics saved in 'evaluation_metrics/evaluation_metrics.xlsx'.")
