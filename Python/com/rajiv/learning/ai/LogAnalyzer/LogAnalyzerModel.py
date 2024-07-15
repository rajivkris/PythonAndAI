import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

torch.backends.mps.is_available = lambda: False

data = pd.read_csv('spring_boot_synthetic_log_data.csv')

label_mapping = {'Backend Team': 0, 'Frontend Team': 1, 'Database Team': 2, 'DevOps Team': 3, 'Security Team': 4}
data['label'] = data['label'].map(label_mapping)

X_train, X_test, y_train, y_test = train_test_split(data['log_message'], data['label'], test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

train_encodings = tokenize_function(X_train.tolist())
test_encodings = tokenize_function(X_test.tolist())

device = torch.device("cpu")
print(f"Using device: {device}")

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))

model.to(device)

class LogDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].to(device) for key, val in self.encodings.items()}  # Move encodings to the device
        item['labels'] = torch.tensor(self.labels.iloc[idx], device=device)  # Move label tensor to the device
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = LogDataset(train_encodings, y_train)
test_dataset = LogDataset(test_encodings, y_test)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

def create_trainer(model, train_dataset, test_dataset, training_args):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

trainer = create_trainer(model, train_dataset, test_dataset, training_args)

trainer.train()
trainer.evaluate()

predictions = trainer.predict(test_dataset)

preds = torch.tensor(predictions.predictions).argmax(dim=-1)

print(classification_report(y_test, preds.cpu()))

model.save_pretrained('./model')
tokenizer.save_pretrained('./model')