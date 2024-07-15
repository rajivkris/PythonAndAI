import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Ensure a GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the saved tokenizer and model
tokenizer = BertTokenizer.from_pretrained('./model')
model = BertForSequenceClassification.from_pretrained('./model', num_labels=5)  # Adjust num_labels if necessary
model.to(device)

new_data = [
    "Token mismatch while parsing json"
]

def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

new_encodings = tokenize_function(new_data)

input_ids = new_encodings['input_ids'].to(device)
attention_mask = new_encodings['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1)

label_mapping = {0: 'Backend Team', 1: 'Frontend Team', 2: 'Database Team', 3: 'DevOps Team', 4: 'Security Team'}

predicted_labels = [label_mapping[pred.item()] for pred in predictions]

for text, label in zip(new_data, predicted_labels):
    print(f"Text: {text}\nPredicted Label: {label}\n")