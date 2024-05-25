# converting the json file to csv

import pandas as pd
import json

# Reading JSON data from a file
with open("/home/hema/files/english.json", 'r', encoding='utf-8') as file:
    json_data = file.readlines()
    # remove the extra data causing issues
    json_data = "[" + ','.join([line.strip() for line in json_data if 'text' in line]) + "]"

# Converting JSON data to a pandas DataFrame
df = pd.read_json(json_data)

# Writing DataFrame to a CSV file
df.to_csv("english.csv", index=False)

# preprocessing the data

# Dropping the required columns
df = df.drop(columns=["did", "uid", "date"])

# Writing the modified DataFrame to a CSV file
df.to_csv("preprocessed_data.csv", index=False)

print(df.columns)

# splitting the data
# 80% for training, 10% for testing and 10% for development.

from sklearn.model_selection import train_test_split

# Load your dataset into a pandas DataFrame
df = pd.read_csv('preprocessed_data.csv')

# Split your data into train, test, and development sets
x_train, x_test_dev, y_train, y_test_dev = train_test_split(df.drop(['text', 'gender', 'age', 'country'], axis=1), df[['text', 'gender', 'age', 'country']], test_size=0.2, random_state=42)

# Further split the test+dev data into test and development sets
x_test, x_dev, y_test, y_dev = train_test_split(x_test_dev, x_test_dev, test_size=0.5, random_state=42)

from sklearn.model_selection import train_test_split

# Split the data into train, validation, and test sets
train, test_dev = train_test_split(df, test_size=0.2, random_state=42)
dev, test = train_test_split(test_dev, test_size=0.5, random_state=42)

# Save train, validation, and test sets as TSV files
train.to_csv('train.tsv', sep='\t', index=False)
dev.to_csv('dev.tsv', sep='\t', index=False)
test.to_csv('test.tsv', sep='\t', index=False)

def load_and_preprocess(data_path, test=False):
    x = []
    y = []
    genders = []  # new list to store gender information

    with open(data_path, encoding='utf8') as dfile:
        cols = dfile.readline().strip().split('\t')

        text_idx = 0
        label_idx = 4
        gender_idx = 1  # index of the column containing gender information

        next(dfile)  # skip the header line

        for line in dfile:
            line = line.strip().split('\t')

            if len(line) < 5:  # check if line has enough elements
                continue

            x.append(line[text_idx])
            y.append(int(round(float(line[label_idx]))))
            genders.append(line[gender_idx])  # extract gender information

    return x, y, genders  # return x, y, and genders


train_x, train_y, train_genders = load_and_preprocess('train.tsv') # training set
dev_x, dev_y, dev_genders = load_and_preprocess('dev.tsv') # development set
test_x, test_y, test_genders = load_and_preprocess('test.tsv', test=True) # test set

# doing the stats
def count_gender_ratio(gender_list):
    male = 0
    female = 0
    not_reported = 0
    for gender in gender_list:
        if gender == 'm':
            male += 1
        elif gender == 'f':
            female += 1
        else:
            not_reported += 1
    ratio_males_to_females = male / female
    print(f'Males count = {male}')
    print(f'Females count = {female}')
    print(f'Not Reported count = {not_reported}')
    print(male + female+not_reported)
    print()
    return ratio_males_to_females

print(count_gender_ratio(train_genders))
print(len(train_genders))

def count_positive_negative_labels(label_list):
    positive = 0
    negative = 0
    for label in label_list:
        if label == 1:
            positive += 1
        if label == 0:
            negative += 1
    ratio_positive_to_negative_labels = positive / negative
    print(f'Positive Label count = {positive}')
    print(f'Negative Label count = {negative}')
    print()
    print(positive + negative)
    return ratio_positive_to_negative_labels

print(count_positive_negative_labels(train_y))
print(len(train_y))

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import time

# Step 1: Data preparation
train_reviews = train_x  # List of product reviews for training
train_labels = train_y  # List of corresponding sentiment labels for training

eval_reviews = test_x  # List of product reviews for evaluation
eval_labels = test_y  # List of corresponding sentiment labels for evaluation

# Step 2: Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_reviews, truncation=True, padding=True, max_length=150)
eval_encodings = tokenizer(eval_reviews, truncation=True, padding=True, max_length=150)

# Step 3: Creating input sequences
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                              torch.tensor(train_encodings['attention_mask']),
                                              torch.tensor(train_labels))

eval_dataset = torch.utils.data.TensorDataset(torch.tensor(eval_encodings['input_ids']),
                                             torch.tensor(eval_encodings['attention_mask']),
                                             torch.tensor(eval_labels))

# Step 4: Training setup

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)  # 2 for binary sentiment classification
optimizer = AdamW(model.parameters(), lr=1e-5)

train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=3, shuffle=False)


# Step 5: Training process
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

total_train_batches = len(train_dataloader)
total_epochs = 1  # Change the number of epochs if desired

for epoch in range(total_epochs):
    start_time = time.time()
    model.train()
    total_train_loss = 0
    total_train_samples = 0
    correct_train_predictions = 0

    for batch_idx, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        _, predicted_labels = torch.max(outputs.logits, 1)
        correct_train_predictions += (predicted_labels == labels).sum().item()

        total_train_samples += labels.size(0)

        loss.backward()
        optimizer.step()

        # Print progress update
        progress = (batch_idx + 1) / total_train_batches * 100
        print(f"Epoch: {epoch + 1} | Batch: {batch_idx + 1}/{total_train_batches} | Progress: {progress:.2f}%")

    model.eval()
    eval_accuracy = 0
    total_eval_samples = 0
    correct_eval_predictions = 0

    for batch in eval_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

        eval_accuracy += torch.sum(predictions == labels).item()
        total_eval_samples += labels.size(0)
        correct_eval_predictions += (predictions == labels).sum().item()

    avg_train_loss = total_train_loss / total_train_batches
    avg_train_accuracy = correct_train_predictions / total_train_samples
    avg_eval_accuracy = correct_eval_predictions / total_eval_samples

    # Print information to monitor the training process
    secs = int(time.time() - start_time)
    mins = secs // 60
    secs = secs % 60
    print(f"Epoch: {epoch + 1} | Time: {mins} minutes, {secs} seconds")
    print(f"\tLoss: {avg_train_loss:.4f} (Train) | Accuracy: {avg_train_accuracy*100:.1f}% (Train)")
    print(f"\tAccuracy: {avg_eval_accuracy*100:.1f}% (Eval)")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

model.eval()

eval_predictions = []
eval_labels = []

for batch in eval_dataloader:
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    eval_predictions.extend(predictions.cpu().tolist())
    eval_labels.extend(labels.cpu().tolist())

classification_report_dict = classification_report(eval_labels, eval_predictions, digits=4, output_dict=True)
precision = classification_report_dict["weighted avg"]["precision"]
recall = classification_report_dict["weighted avg"]["recall"]
f1 = classification_report_dict["weighted avg"]["f1-score"]
accuracy = (sum(eval_labels[i] == eval_predictions[i] for i in range(len(eval_labels))) / len(eval_labels)) * 100
roc_auc = roc_auc_score(eval_labels, eval_predictions)

print(f"Precision: {float(precision):.4f}")
print(f"Accuracy: {float(accuracy):.2f}%")
print(f"Recall: {float(recall):.4f}")
print(f"ROC-AUC score: {float(roc_auc):.4f}")
print(f"F1 score: {float(f1):.4f}")
print()

# Additional print statements for detailed information
print("Classification Report:")
print(classification_report(eval_labels, eval_predictions))
print()

# this confusion matrix is for both the genders in the test data
print("Confusion Matrix:")
print(confusion_matrix(eval_labels, eval_predictions))
print()

male_x_reviews = []
male_y_labels = []
female_x_reviews = []
female_y_labels = []

# Loop 1: Separate male data
for i in range(len(test_x)):
    if test_genders[i] == "m":
        male_x_reviews.append(test_x[i])
        male_y_labels.append(test_y[i])

# Loop 2: Separate female data
for i in range(len(test_x)):
    if test_genders[i] == "f":
        female_x_reviews.append(test_x[i])
        female_y_labels.append(test_y[i])

# Loop 3: Print data lengths
male_data_length = len(male_x_reviews)
female_data_length = len(female_x_reviews)
print("Male data length:", male_data_length)
print("Female data length:", female_data_length)

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import time



eval_reviews = male_x_reviews  # List of product reviews for evaluation
eval_labels = male_y_labels  # List of corresponding sentiment labels for evaluation

# Step 2: Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_reviews, truncation=True, padding=True, max_length=150)
eval_encodings = tokenizer(eval_reviews, truncation=True, padding=True, max_length=150)

# Step 3: Creating input sequences

eval_dataset = torch.utils.data.TensorDataset(torch.tensor(eval_encodings['input_ids']),
                                             torch.tensor(eval_encodings['attention_mask']),
                                             torch.tensor(eval_labels))

# Step 4: Training setup

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)  # 2 for binary sentiment classification
optimizer = AdamW(model.parameters(), lr=1e-5)

train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=3, shuffle=False)

# Step 5: Training process
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

model.eval()
eval_accuracy = 0
total_eval_samples = 0
correct_eval_predictions = 0

for batch in eval_dataloader:
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    eval_accuracy += torch.sum(predictions == labels).item()
    total_eval_samples += labels.size(0)
    correct_eval_predictions += (predictions == labels).sum().item()

avg_train_loss = total_train_loss / total_train_batches
avg_train_accuracy = correct_train_predictions / total_train_samples
avg_eval_accuracy = correct_eval_predictions / total_eval_samples

# Print information to monitor the training process
secs = int(time.time() - start_time)
mins = secs // 60
secs = secs % 60
print(f"Epoch: {epoch + 1} | Time: {mins} minutes, {secs} seconds")
print(f"\tLoss: {avg_train_loss:.4f} (Train) | Accuracy: {avg_train_accuracy*100:.1f}% (Train)")
print(f"\tAccuracy: {avg_eval_accuracy*100:.1f}% (Eval)")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

model.eval()

eval_predictions = []
eval_labels = []

for batch in eval_dataloader:
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    eval_predictions.extend(predictions.cpu().tolist())
    eval_labels.extend(labels.cpu().tolist())

classification_report_dict = classification_report(eval_labels, eval_predictions, digits=4, output_dict=True)
precision = classification_report_dict["weighted avg"]["precision"]
recall = classification_report_dict["weighted avg"]["recall"]
f1 = classification_report_dict["weighted avg"]["f1-score"]
accuracy = (sum(eval_labels[i] == eval_predictions[i] for i in range(len(eval_labels))) / len(eval_labels)) * 100
roc_auc = roc_auc_score(eval_labels, eval_predictions)

print(f"Precision: {float(precision):.4f}")
print(f"Accuracy: {float(accuracy):.2f}%")
print(f"Recall: {float(recall):.4f}")
print(f"ROC-AUC score: {float(roc_auc):.4f}")
print(f"F1 score: {float(f1):.4f}")
print()

# Additional print statements for detailed information
print("Classification Report:")
print(classification_report(eval_labels, eval_predictions))
print()

# this confusion matrix is for both the genders in the test data
print("Confusion Matrix:")
print(confusion_matrix(eval_labels, eval_predictions))
print()

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import time


eval_reviews = female_x_reviews  # List of product reviews for evaluation
eval_labels = female_y_labels  # List of corresponding sentiment labels for evaluation

# Step 2: Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_reviews, truncation=True, padding=True, max_length=150)
eval_encodings = tokenizer(eval_reviews, truncation=True, padding=True, max_length=150)

# Step 3: Creating input sequences

eval_dataset = torch.utils.data.TensorDataset(torch.tensor(eval_encodings['input_ids']),
                                             torch.tensor(eval_encodings['attention_mask']),
                                             torch.tensor(eval_labels))

# # Step 4: Training setup

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)  # 2 for binary sentiment classification
optimizer = AdamW(model.parameters(), lr=1e-5)

train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=3, shuffle=False)


# Step 5: Training process

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

model.eval()
eval_accuracy = 0
total_eval_samples = 0
correct_eval_predictions = 0

for batch in eval_dataloader:
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    eval_accuracy += torch.sum(predictions == labels).item()
    total_eval_samples += labels.size(0)
    correct_eval_predictions += (predictions == labels).sum().item()

avg_train_loss = total_train_loss / total_train_batches
avg_train_accuracy = correct_train_predictions / total_train_samples
avg_eval_accuracy = correct_eval_predictions / total_eval_samples

# Print information to monitor the training process
secs = int(time.time() - start_time)
mins = secs // 60
secs = secs % 60
print(f"Epoch: {epoch + 1} | Time: {mins} minutes, {secs} seconds")
print(f"\tLoss: {avg_train_loss:.4f} (Train) | Accuracy: {avg_train_accuracy*100:.1f}% (Train)")
print(f"\tAccuracy: {avg_eval_accuracy*100:.1f}% (Eval)")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

model.eval()

eval_predictions = []
eval_labels = []

for batch in eval_dataloader:
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    eval_predictions.extend(predictions.cpu().tolist())
    eval_labels.extend(labels.cpu().tolist())

classification_report_dict = classification_report(eval_labels, eval_predictions, digits=4, output_dict=True)
precision = classification_report_dict["weighted avg"]["precision"]
recall = classification_report_dict["weighted avg"]["recall"]
f1 = classification_report_dict["weighted avg"]["f1-score"]
accuracy = (sum(eval_labels[i] == eval_predictions[i] for i in range(len(eval_labels))) / len(eval_labels)) * 100
roc_auc = roc_auc_score(eval_labels, eval_predictions)

print(f"Precision: {float(precision):.4f}")
print(f"Accuracy: {float(accuracy):.2f}%")
print(f"Recall: {float(recall):.4f}")
print(f"ROC-AUC score: {float(roc_auc):.4f}")
print(f"F1 score: {float(f1):.4f}")
print()

# Additional print statements for detailed information
print("Classification Report:")
print(classification_report(eval_labels, eval_predictions))
print()

# this confusion matrix is for both the genders in the test data
print("Confusion Matrix:")
print(confusion_matrix(eval_labels, eval_predictions))
print()

