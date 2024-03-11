import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
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



if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (Nvidia GPU).")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU.")

model = DistilBertForSequenceClassification.from_pretrained('./pardonmyai')
model.to(device)


# Load the data

input_ids_file = 'input_ids.pt'
attention_masks_file = 'attention_masks.pt'
labels_file = 'labels.pt'

if os.path.exists(input_ids_file) and os.path.exists(attention_masks_file) and os.path.exists(labels_file):
    print("Loading tokenized data from saved files.")
    input_ids = torch.load(input_ids_file)
    attention_masks = torch.load(attention_masks_file)
    labels = torch.load(labels_file)
else:

    df = pd.read_csv("clean_data.csv")
    dataset = Dataset.from_pandas(df)


    def filter_empty_texts(example):
        return example["text"] is not None and example["text"].strip() != ""


    dataset = dataset.filter(filter_empty_texts)
    dataset = dataset.rename_column("is_offensive", "label")

    texts = df['text'].tolist()
    labels = df['is_offensive'].tolist()

    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('./pardonmyai')

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
    torch.save(input_ids, input_ids_file)
    torch.save(attention_masks, attention_masks_file)
    torch.save(labels, labels_file)


batch_size = 32

# Create the DataLoader
prediction_data = TensorDataset(input_ids, attention_masks, labels)

prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Evaluation
print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

model.eval()
predictions, true_labels = [], []

for batch in tqdm(prediction_dataloader, "predicting"):
    batch = tuple(t.to(device) for t in batch)
    
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)
    
    logits = outputs[0].detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    predictions.extend(logits)
    true_labels.extend(label_ids)

# Convert predictions to 0 or 1
pred_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = np.array(true_labels).flatten()

# Calculate the evaluation metrics
accuracy = accuracy_score(labels_flat, pred_flat)
precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, pred_flat, average='binary')
auc_roc = roc_auc_score(labels_flat, [p[1] for p in predictions])

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC-ROC: {auc_roc:.4f}')
