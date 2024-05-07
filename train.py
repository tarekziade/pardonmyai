"""
Fine-tune a Distilbert model on a profanity dataset.
"""
from codecarbon import track_emissions
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
from transformers.trainer_callback import EarlyStoppingCallback


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


def get_datasets(args, tokenizer):
    if args.fine_tune:
        dataset = load_dataset("tarekziade/animal_descriptions", split="train")
    else:
        dataset = load_dataset("tarekziade/profanity-clean", split="train")

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
    output_dir="pardonmycaption",
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


@track_emissions(project_name="PardonMyAI")
def train(args):
    if args.tiny:
        model_name = TINY_BASE_MODEL
        model_path = MODEL_PATH + "-tiny"
        tokenizer_klass = BertTokenizer
        model_klass = BertForSequenceClassification
    elif args.fine_tune:
        model_name = "tarekziade/pardonmyai"
        model_path = MODEL_PATH + "-fine-tune"
        tokenizer_klass = DistilBertTokenizer
        model_klass = DistilBertForSequenceClassification
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

    datasets = get_datasets(args, tokenizer)

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    try:
        trainer.train()
    finally:
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    trainer.push_to_hub("tarekziade/pardonmycaption")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the profanity model")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Enable the tiny mode.",
        default=False,
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Second pass",
        default=False,
    )

    args = parser.parse_args()

    train(args)
