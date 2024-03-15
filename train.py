"""
Fine-tune a Distilbert model on a profanity dataset.
"""
import argparse
import torch
import evaluate
from datasets import load_dataset
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import (
    DistilBertTokenizer,
    BertTokenizer,
    DistilBertForSequenceClassification,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


BASE_MODEL = "distilbert-base-uncased"
TINY_BASE_MODEL = "huawei-noah/TinyBERT_General_4L_312D"
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


def get_datasets(tokenizer):
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


def train(tiny=False):
    if tiny:
        model_name = TINY_BASE_MODEL
        model_path = MODEL_PATH + "-tiny"
        tokenizer_klass = BertTokenizer
        model_klass = BertForSequenceClassification
    else:
        model_name = BASE_MODEL
        model_path = MODEL_PATH
        tokenizer_klass = DistilBertTokenizer
        model_klass = DistilBertForSequenceClassification

    tokenizer = tokenizer_klass.from_pretrained(model_name)
    model = model_klass.from_pretrained(
        model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    model.to(device)

    datasets = get_datasets(tokenizer)

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
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the profanity model")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Enable the tiny mode.",
    )
    args = parser.parse_args()

    train(args.tiny)
