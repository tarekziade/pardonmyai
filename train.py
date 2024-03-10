import torch
import evaluate
import pandas as pd
from datasets import Dataset
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


accuracy = evaluate.load("accuracy")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (Nvidia GPU).")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# Load the data
df = pd.read_csv("clean_data.csv")
dataset = Dataset.from_pandas(df)


def filter_empty_texts(example):
    return example["text"] is not None and example["text"].strip() != ""


dataset = dataset.filter(filter_empty_texts)
dataset = dataset.rename_column("is_offensive", "label")


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


id2label = {0: "NOT_OFFENSIVE", 1: "OFFENSIVE"}
label2id = {"NOT_OFFENSIVE": 0, "OFFENSIVE": 1}


# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and testing
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)

# Load the model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)
model.to(device)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    tokenizer=tokenizer,
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,  # You can define a compute_metrics function for evaluation
)

# Train the model
trainer.train()

# save it
model.save_pretrained("./pardonmyai")
tokenizer.save_pretrained("./pardonmyai")
