"""
This script converts the original CSV file to a clean JSON file, and keep only small examples.
"""
import re

import pandas as pd
from datasets import Dataset


def clean_text(text):
    if text is None:
        return ""
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def clean_ds(entries):
    entries["text"] = [clean_text(text) for text in entries["text"]]
    return entries


def keep(entries):
    def check(text):
        return text.strip() != "" and len(text) < 150

    return [check(text) for text in entries["text"]]


if __name__ == "__main__":
    df = pd.read_csv("clean_data.csv")
    ds = Dataset.from_pandas(df)
    ds = ds.map(clean_ds, batched=True)
    ds = ds.filter(keep, batched=True)
    ds.to_json("profanity-clean.json")
    ds.push_to_hub("tarekziade/profanity-clean")
