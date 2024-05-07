"""
Infere the text passed in the CLI
"""
from transformers import pipeline


tests = [
    ("An obese woman playing with her dog in the park", "OFFENSIVE"),
    ("A black cat liking milk", "NOT_OFFENSIVE"),
    ("A fat man playing with her dog in the park", "OFFENSIVE"),
    ("A black cow eating grass", "NOT_OFFENSIVE"),
    ("An ugly kitten eating food", "OFFENSIVE"),
]


if __name__ == "__main__":
    print("Loading model...")
    classifier = pipeline("sentiment-analysis", model="tarekziade/pardonmycaption")

    for sentence, expected in tests:
        print(f"Testing: {sentence}")
        res = classifier(sentence)
        assert res[0]["label"] == expected
