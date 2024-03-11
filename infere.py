import random 
import time
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)

offensive_sentences_with_cuss_words = [
    "there is a man with a big black cock on his butt",
    "You're such an ass.",
    "That's a cunt idea.",
    "What a fucked up thing to do.",
    "I hate your fucking guts.",
    "You're a shithead.",
    "This is bullshit.",
    "Don't be an ass.",
    "That's shit.",
    "What the hell?",
    "Screw this."
]


sentences = [
    "The sky is clear today.",
    "What a beautiful garden!",
    "I really enjoy reading books.",
    "The cake was absolutely delicious.",
    "He plays the guitar so well.",
    "She has a lovely singing voice.",
    "Let's go for a walk in the park.",
    "I had a great time at the party.",
    "This painting is stunning.",
    "Can you help me with this problem?",
    "I've always wanted to visit Japan.",
    "The movie last night was really scary.",
    "I love how peaceful it is in the morning.",
    "That was a very interesting lecture.",
    "The new restaurant in town is amazing.",
    "I can't wait for the weekend.",
    "The book you recommended was fantastic.",
    "I'm really proud of my achievements.",
    "The sunset was breathtakingly beautiful.",
    "Your new haircut looks great."
]

# Offensive sentences
offensive_sentences = [
    "You're such an idiot.",
    "That's a stupid idea.",
    "What a dumb thing to do.",
    "I hate your guts.",
    "You're a loser."
]

def generate_sentences(sentences, offensive_sentences, total=500, offensive_percentage=0.1):
    num_offensive = int(total * offensive_percentage)
    all_sentences = random.sample(sentences * (total // len(sentences) + 1), total - num_offensive)
    all_sentences += random.sample(offensive_sentences * (num_offensive // len(offensive_sentences) + 1), num_offensive)

    random.shuffle(all_sentences)

    sentence_list = []
    for sentence in all_sentences:
        sentence_list.append({
            "text": sentence,
            "is_offensive": sentence in offensive_sentences
        })
    
    return sentence_list

sentence_list_with_cuss_words = generate_sentences(sentences, offensive_sentences_with_cuss_words)
texts= [sentence["text"] for sentence in sentence_list_with_cuss_words]


tokenizer = DistilBertTokenizer.from_pretrained("./pardonmyai")
model = DistilBertForSequenceClassification.from_pretrained("./pardonmyai")

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

start = time.time()
with torch.no_grad():
    outputs = model(**inputs)
print(f"Duration {time.time() - start}")

predictions = torch.argmax(outputs.logits, dim=1)

labels = ["Not Offensive", "Offensive"]
predicted_labels = [labels[prediction] for prediction in predictions]

for text, predicted_label in zip(texts, predicted_labels):
    if predicted_label == "Offensive":
        print(f"Text: {text}\nPredicted label: {predicted_label}\n")
