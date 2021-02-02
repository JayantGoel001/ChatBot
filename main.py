import random
import json
import pickle as pkl
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = {}
with open("intents.json", "r+") as f:
    intents = json.loads(f.read())

words = []
with open("words.pkl", 'r+') as f:
    words = pkl.load(f)

classes = []
with open("classes.pkl", 'r+') as f:
    classes = pkl.load(f)

model = load_model("chatbot")


def clean_up_sentence(sentence):
    sentence_word = nltk.word_tokenize(sentence)
    sentence_word = [lemmatizer.lemmatize(word) for word in sentence_word]
    return sentence_word


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict(sentence):
    bag = bag_of_words(sentence)
    res = model.predict(np.array(bag))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)

    result_list = []
