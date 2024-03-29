"""
Infere the text passed in the CLI
"""
import sys
import time
from transformers import pipeline


if __name__ == "__main__":
    print("Loading model...")
    classifier = pipeline("sentiment-analysis", model="./pardonmyai")

    start = time.time()
    res = classifier(sys.argv[-1])
    print(f"Inference took {time.time() - start:.2f} seconds.")
    print(res)
