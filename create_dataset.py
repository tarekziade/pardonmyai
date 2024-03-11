"""
This script converts the original CSV file to a clean JSON file.
"""
import pandas as pd
from datasets import Dataset


df = pd.read_csv("clean_data.csv")
dataset = Dataset.from_pandas(df)


def filter_empty_texts(example):
    return example["text"] is not None and example["text"].strip() != ""


dataset = dataset.filter(filter_empty_texts)

dataset.to_json("profanity-check.json")
