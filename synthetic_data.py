import re
from datasets import Dataset, concatenate_datasets
import pandas as pd
import random

animals = ["cow"] * 10 + [
    "dog",
    "cat",
    "horse",
    "rabbit",
    "sheep",
    "fox",
    "parakeet",
    "iguana",
    "canary",
    "hamster",
    "fish",
    "pig",
    "bunny",
    "bear",
    "squirrel",
    "kangaroo",
    "chick",
    "parrot",
    "goat",
    "camel",
    "poodle",
    "polar bear",
    "starling",
    "goldfish",
    "lion",
    "donkey",
    "pigeon",
    "peacock",
    "labrador",
    "wolf",
    "swan",
    "crocodile",
    "pheasant",
    "cardinal",
    "duck",
    "dove",
    "raven",
    "jay",
    "turtle",
    "deer",
    "elephant",
    "giraffe",
    "mongoose",
    "zebra",
    "owl",
    "frog",
    "mouse",
    "vulture",
    "bat",
]
offensive_adjectives = [
    "ugly",
    "stupid",
    "fat",
    "smelly",
    "lazy",
    "dumb",
    "dirty",
    "sick",
    "crazy",
    "obese",
]

subjects = ["man", "woman", "child", "boy", "girl"]

adjectives = [
    "black",
    "white",
    "brown",
    "grey",
    "spotted",
    "golden",
    "striped",
    "red",
    "blue",
    "green",
    "yellow",
    "tan",
    "silver",
    "pink",
    "orange",
    "cream",
    "dark brown",
    "light grey",
    "rusty red",
    "pale yellow",
    "turquoise",
    "beige",
    "deep purple",
    "sandy",
    "charcoal grey",
    "navy blue",
    "lavender",
    "chocolate",
    "olive green",
    "crimson",
    "teal",
    "pale grey",
    "jet black",
    "sapphire",
    "deep green",
    "pearl white",
    "copper",
    "lemon yellow",
    "ruby red",
    "chestnut",
]
actions = [
    "running",
    "sleeping",
    "grazing",
    "hopping",
    "standing",
    "playing",
    "prowling",
    "resting",
    "sneaking",
    "singing",
    "sunning",
    "chirping",
    "spinning",
    "swimming",
    "wallowing",
    "stalking",
    "nibbling",
    "eating",
    "fishing",
    "gathering",
    "hopping",
    "pecking",
    "mimicking",
    "climbing",
    "prancing",
    "walking",
    "perched",
    "floating",
    "lounging",
    "braying",
    "cooing",
    "displaying",
    "playing",
    "resting",
    "howling",
    "gliding",
    "lurking",
    "strutting",
    "flitting",
    "dabbling",
    "perched",
    "cawing",
    "squawking",
    "crawling",
    "purring",
    "darting",
    "slithering",
    "hovering",
    "trotting",
]
settings = [
    "in the park",
    "on a sunny windowsill",
    "in a meadow",
    "through a garden",
    "near a barn",
    "by the lake",
    "in the backyard",
    "under a tree",
    "through the woods",
    "in a cage",
    "on a rock",
    "in a pet shop",
    "on a wheel",
    "in a clear pond",
    "in the mud",
    "at a butterfly",
    "on some carrots",
    "eating bamboo",
    "in a river",
    "in a park",
    "in an open field",
    "on a farm",
    "in the living room",
    "on a small hill",
    "in the desert sun",
    "during a dog show",
    "across the ice",
    "on a tree branch",
    "in a tank",
    "in the savannah",
    "in a field",
    "on a city street",
    "its feathers",
    "with a ball of yarn",
    "by the fireplace",
    "at the moon",
    "across a lake",
    "in the swamp",
    "through the underbrush",
    "from tree to tree",
    "in a pond",
    "peacefully on a branch",
    "atop a fence",
    "in a tree",
    "on the beach",
    "on a soft pillow",
    "through the underbrush",
    "in the grass",
    "near a feeder",
    "along a trail",
]


def generate_descriptions(num_ok, num_offensive):
    descriptions = list()
    while len(descriptions) < num_ok:
        animal = random.choice(animals)
        adjective = random.choice(adjectives)
        action = random.choice(actions)
        setting = random.choice(settings)
        description = f"A {adjective} {animal} is {action} {setting}."
        descriptions.append({"text": description, "is_offensive": 0})

    while len(descriptions) < (num_ok + num_offensive):
        subject = random.choice(subjects)
        adjective = random.choice(offensive_adjectives)
        action = random.choice(actions)
        setting = random.choice(settings)
        description = f"A {adjective} {subject} is {action} {setting}."
        descriptions.append({"text": description, "is_offensive": 1})

    return descriptions


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

    descriptions = generate_descriptions(1700, 300)

    data = {
        "text": [e["text"] for e in descriptions],
        "is_offensive": [e["is_offensive"] for e in descriptions],
    }
    df = pd.DataFrame(data)
    ds2 = Dataset.from_pandas(df)

    combined = concatenate_datasets([ds, ds2])

    combined.shuffle(42)
    combined.push_to_hub("tarekziade/pardonmyai-clean")
