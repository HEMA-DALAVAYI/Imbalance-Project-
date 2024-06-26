import pandas as pd
import json
from sklearn.model_selection import train_test_split
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, Dataset
import time
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Convert the JSON file to CSV and preprocess the data
with open("/home/hema/files/english.json", 'r', encoding='utf-8') as file:
    json_data = file.readlines()
    json_data = "[" + ','.join([line.strip() for line in json_data if 'text' in line]) + "]"

df = pd.read_json(json_data)
df.to_csv("english.csv", index=False)

df = df.drop(columns=["did", "uid", "date"])
df.to_csv("preprocessed_data.csv", index=False)

# 2. Split the data into training, validation, and test sets
df = pd.read_csv('preprocessed_data.csv')
train, test_dev = train_test_split(df, test_size=0.2, random_state=42)
dev, test = train_test_split(test_dev, test_size=0.5, random_state=42)

train.to_csv('train.tsv', sep='\t', index=False)
dev.to_csv('dev.tsv', sep='\t', index=False)
test.to_csv('test.tsv', sep='\t', index=False)

# 3. Load and preprocess the data, including gender information
def load_and_preprocess(data_path, test=False):
    x = []
    y = []
    genders = []

    with open(data_path, encoding='utf8') as dfile:
        cols = dfile.readline().strip().split('\t')

        text_idx = 0
        label_idx = 4
        gender_idx = 1

        next(dfile)

        for line in dfile:
            line = line.strip().split('\t')
            if len(line) < 5:
                continue

            x.append(line[text_idx])
            y.append("positive" if int(round(float(line[label_idx]))) == 1 else "negative")
            genders.append(line[gender_idx])

    return x, y, genders

train_x, train_y, train_genders = load_and_preprocess('train.tsv')
dev_x, dev_y, dev_genders = load_and_preprocess('dev.tsv')
test_x, test_y, test_genders = load_and_preprocess('test.tsv', test=True)

# 4. Create a custom dataset class for T5
class T5Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = ["sentiment analysis: " + text for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        label_encoding = self.tokenizer(
            label,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = label_encoding['input_ids'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# 5. Train the T5 model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
train_dataset = T5Dataset(train_x, train_y, tokenizer, max_length=128)
eval_dataset = T5Dataset(test_x, test_y, tokenizer, max_length=128)

model = T5ForConditionalGeneration.from_pretrained('t5-small')
optimizer = AdamW(model.parameters(), lr=1e-5)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

total_train_batches = len(train_dataloader)
total_epochs = 1

for epoch in range(total_epochs):
    start_time = time.time()
    model.train()
    total_train_loss = 0
    total_train_samples = 0

    for batch_idx, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        total_train_samples += input_ids.size(0)

        loss.backward()
        optimizer.step()

        progress = (batch_idx + 1) / total_train_batches * 100
        print(f"Epoch: {epoch + 1} | Batch: {batch_idx + 1}/{total_train_batches} | Progress: {progress:.2f}%")

    model.eval()
    eval_accuracy = 0
    total_eval_samples = 0
    correct_eval_predictions = 0
    eval_predictions = []
    eval_labels = []

    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        eval_predictions.extend(predictions)
        eval_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

    correct_eval_predictions = sum(p == l for p, l in zip(eval_predictions, eval_labels))
    total_eval_samples = len(eval_labels)
    avg_train_loss = total_train_loss / total_train_batches
    avg_eval_accuracy = correct_eval_predictions / total_eval_samples

    secs = int(time.time() - start_time)
    mins = secs // 60
    secs = secs % 60
    print(f"Epoch: {epoch + 1} | Time: {mins} minutes, {secs} seconds")
    print(f"\tLoss: {avg_train_loss:.4f} (Train) | Accuracy: {avg_eval_accuracy*100:.1f}% (Eval)")

# 6. Evaluate the model on the overall test set
classification_report_dict = classification_report(eval_labels, eval_predictions, digits=4, output_dict=True)
precision = classification_report_dict["weighted avg"]["precision"]
recall = classification_report_dict["weighted avg"]["recall"]
f1 = classification_report_dict["weighted avg"]["f1-score"]
accuracy = (correct_eval_predictions / total_eval_samples) * 100
roc_auc = roc_auc_score([1 if l == "positive" else 0 for l in eval_labels], [1 if p == "positive" else 0 for p in eval_predictions])

print(f"Precision: {float(precision):.4f}")
print(f"Accuracy: {float(accuracy):.2f}%")
print(f"Recall: {float(recall):.4f}")
print(f"ROC-AUC score: {float(roc_auc):.4f}")
print(f"F1 score: {float(f1):.4f}")

print("Classification Report:")
print(classification_report(eval_labels, eval_predictions))

print("Confusion Matrix:")
print(confusion_matrix(eval_labels, eval_predictions))

# 7. Separate male and female test data and perform evaluations for each
male_x_reviews = [test_x[i] for i in range(len(test_x)) if test_genders[i] == "m"]
male_y_labels = ["positive" if y == 1 else "negative" for i, y in enumerate(test_y) if test_genders[i] == "m"]

female_x_reviews = [test_x[i] for i in range(len(test_x)) if test_genders[i] == "f"]
female_y_labels = ["positive" if y == 1 else "negative" for i, y in enumerate(test_y) if test_genders[i] == "f"]

male_eval_dataset = T5Dataset(male_x_reviews, male_y_labels, tokenizer, max_length=128)
female_eval_dataset = T5Dataset(female_x_reviews, female_y_labels, tokenizer, max_length=128)

male_eval_dataloader = DataLoader(male_eval_dataset, batch_size=16, shuffle=False)
female_eval_dataloader = DataLoader(female_eval_dataset, batch_size=16, shuffle=False)

# Evaluate on male test data
model.eval()
male_eval_predictions = []
male_eval_labels = []

for batch in male_eval_dataloader:
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    male_eval_predictions.extend(predictions)
    male_eval_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

male_classification_report_dict = classification_report(male_eval_labels, male_eval_predictions, digits=4, output_dict=True)
male_precision = male_classification_report_dict["weighted avg"]["precision"]
male_recall = male_classification_report_dict["weighted avg"]["recall"]
male_f1 = male_classification_report_dict["weighted avg"]["f1-score"]
male_accuracy = (sum(p == l for p, l in zip(male_eval_labels, male_eval_predictions)) / len(male_eval_labels)) * 100
male_roc_auc = roc_auc_score([1 if l == "positive" else 0 for l in male_eval_labels], [1 if p == "positive" else 0 for p in male_eval_predictions])

print("\nMale Evaluation:")
print(f"Precision: {float(male_precision):.4f}")
print(f"Accuracy: {float(male_accuracy):.2f}%")
print(f"Recall: {float(male_recall):.4f}")
print(f"ROC-AUC score: {float(male_roc_auc):.4f}")
print(f"F1 score: {float(male_f1):.4f}")

print("Classification Report (Male):")
print(classification_report(male_eval_labels, male_eval_predictions))

print("Confusion Matrix (Male):")
print(confusion_matrix(male_eval_labels, male_eval_predictions))

# Evaluate on female test data
model.eval()
female_eval_predictions = []
female_eval_labels = []

for batch in female_eval_dataloader:
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    female_eval_predictions.extend(predictions)
    female_eval_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

female_classification_report_dict = classification_report(female_eval_labels, female_eval_predictions, digits=4, output_dict=True)
female_precision = female_classification_report_dict["weighted avg"]["precision"]
female_recall = female_classification_report_dict["weighted avg"]["recall"]
female_f1 = female_classification_report_dict["weighted avg"]["f1-score"]
female_accuracy = (sum(p == l for p, l in zip(female_eval_labels, female_eval_predictions)) / len(female_eval_labels)) * 100
female_roc_auc = roc_auc_score([1 if l == "positive" else 0 for l in female_eval_labels], [1 if p == "positive" else 0 for p in female_eval_predictions])

print("\nFemale Evaluation:")
print(f"Precision: {float(female_precision):.4f}")
print(f"Accuracy: {float(female_accuracy):.2f}%")
print(f"Recall: {float(female_recall):.4f}")
print(f"ROC-AUC score: {float(female_roc_auc):.4f}")
print(f"F1 score: {float(female_f1):.4f}")

print("Classification Report (Female):")
print(classification_report(female_eval_labels, female_eval_predictions))

print("Confusion Matrix (Female):")
print(confusion_matrix(female_eval_labels, female_eval_predictions))
