"""
Fine-tune a Distilbert model on a profanity dataset.
"""
import torch
import evaluate
from datasets import load_dataset
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


BASE_MODEL = "distilbert-base-uncased"
DATASET = "tarekziade/profanity"
ID2LABEL = {0: "NOT_OFFENSIVE", 1: "OFFENSIVE"}
LABEL2ID = {"NOT_OFFENSIVE": 0, "OFFENSIVE": 1}
MODEL_PATH = "./pardonmyai"


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


def get_datasets():
    dataset = load_dataset("tarekziade/profanity", split="train")
    dataset = dataset.rename_column("is_offensive", "label")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets.train_test_split(test_size=0.2)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


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


def train():
    tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    model.to(device)

    datasets = get_datasets()

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        compute_metrics=compute_metrics,  # You can define a compute_metrics function for evaluation
    )

    try:
        trainer.train()
    finally:
        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)


if __name__ == "__main__":
    train()
