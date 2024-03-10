import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("./pardonmyai")

texts = ["This is an example sentence.", "Fuck you."]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=1)

labels = ["Not Offensive", "Offensive"]
predicted_labels = [labels[prediction] for prediction in predictions]

for text, predicted_label in zip(texts, predicted_labels):
    print(f"Text: {text}\nPredicted label: {predicted_label}\n")
