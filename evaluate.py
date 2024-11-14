import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

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

# Evaluate the model
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).long()

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# Accuracy
accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Multi-class Precision, Recall, F1 Score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average='weighted')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Create folder for saving diagrams if it doesn't exist
output_dir = "evaluation_diagrams"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Precision-Recall Curve (One-vs-Rest for multi-class)
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

precision_vals, recall_vals, _ = precision_recall_curve(y_test_bin.ravel(), outputs.numpy().ravel())
plt.figure()
plt.plot(recall_vals, precision_vals, color='darkorange', label=f'Precision-Recall curve (area = {precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (One-vs-Rest)')
plt.legend(loc='lower left')
plt.savefig(os.path.join(output_dir, 'precision_recall_curve_multiclass.png'))
plt.close()

# ROC Curve (One-vs-Rest for multi-class)
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), outputs.numpy().ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve_multiclass.png'))
plt.close()

# Save all metrics to a text file
with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
    f.write(f'Accuracy: {accuracy * 100:.2f}%\n')
    f.write(f'Precision: {precision:.2f}\n')
    f.write(f'Recall: {recall:.2f}\n')
    f.write(f'F1 Score: {f1:.2f}\n')
    f.write(f'ROC AUC: {roc_auc:.2f}\n')

print("Evaluation complete. Diagrams saved in 'evaluation_diagrams'.")

# **New Diagrams**

# Loss Curve (assuming you have a history of loss values during training)
# You can plot loss over training epochs
loss_values = np.random.random(200000)  # Replace with actual training loss values
plt.plot(range(1, 200001), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()

# Accuracy vs Epochs
accuracy_values = np.random.random(200000)  # Replace with actual training accuracy values
plt.plot(range(1, 200001), accuracy_values)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.savefig(os.path.join(output_dir, 'accuracy_vs_epochs.png'))
plt.close()

# Feature Importance (using RandomForestClassifier)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_test, y_test)

feature_importances = rf.feature_importances_
sorted_idx = np.argsort(feature_importances)

# Plot feature importances
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), np.array(all_words)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance')
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()
