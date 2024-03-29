"""Script to generate metrics.

Accuracy - measures the proportion of true results (both true positives and true negatives) in the data.
Precision - ratio of true positive predictions to the total predicted positives.
Recall - measures the ability of the model to capture available positives.
F1 Score - harmonic mean of precision and recall, providing a balance between the two.
AUC-ROC - reflects the model's ability to discriminate between the classes.
"""
import os
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
)
import argparse


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (Nvidia GPU).")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU.")


TOKENIZER_NAME = MODEL_NAME = "./pardonmyai"
INPUT_IDS_FILE = "input_ids.pt"
ATTENTION_MASKS_FILE = "attention_masks.pt"
LABELS_FILE = "labels.pt"
BATCH_SIZE = 32


def get_tensors():
    """Returns a possibly cached tuple of tensors"""
    if (
        os.path.exists(INPUT_IDS_FILE)
        and os.path.exists(ATTENTION_MASKS_FILE)
        and os.path.exists(LABELS_FILE)
    ):
        print("Loading tokenized data from saved files.")
        input_ids = torch.load(INPUT_IDS_FILE)
        attention_masks = torch.load(ATTENTION_MASKS_FILE)
        labels = torch.load(LABELS_FILE)
    else:
        dataset = load_dataset("tarekziade/profanity", split="train")
        dataset = dataset.rename_column("is_offensive", "label")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        dataset = dataset.map(tokenize_function, batched=True)
        input_ids = torch.tensor(dataset["input_ids"])
        attention_masks = torch.tensor(dataset["attention_mask"])
        labels = torch.tensor(dataset["label"])

        print("Saving tokenized data for future use.")
        torch.save(input_ids, INPUT_IDS_FILE)
        torch.save(attention_masks, ATTENTION_MASKS_FILE)
        torch.save(labels, LABELS_FILE)
    return input_ids, attention_masks, labels


def compute_metrics(tiny=False):
    if tiny:
        model_name = MODEL_NAME + "-tiny"
    else:
        model_name = MODEL_NAME

    # Create the DataLoader
    prediction_data = TensorDataset(*get_tensors())
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE
    )

    # Evaluation
    print("Loading model")
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    predictions, true_labels = [], []
    for batch in tqdm(prediction_dataloader, "predicting"):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0].detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        predictions.extend(logits)
        true_labels.extend(label_ids)

    # Convert predictions to 0 or 1
    pred_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = np.array(true_labels).flatten()

    # Calculate the evaluation metrics
    accuracy = accuracy_score(labels_flat, pred_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, pred_flat, average="binary"
    )
    auc_roc = roc_auc_score(labels_flat, [p[1] for p in predictions])
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
    }


def eval(tiny=False):
    m = compute_metrics(tiny=tiny)

    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall: {m['recall']:.4f}")
    print(f"F1 Score: {m['f1']:.4f}")
    print(f"AUC-ROC: {m['auc_roc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the profanity model")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Enable the tiny mode.",
    )
    args = parser.parse_args()
    eval(args.tiny)
