# import re
import csv
from template import NaiveBayesClassifier
import string
import pandas as pd
import collections
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# Download the punkt tokenizer if not already downloaded
nltk.download('punkt')
# Download the stopwords if not already downloaded
nltk.download('stopwords')


def preprocess(tweet_string):
    if isinstance(tweet_string, str):
        word_list = word_tokenize(tweet_string)
        stop_words = set(stopwords.words('english'))
        filtered_word_list = [word.lower() for word in word_list if word.lower() not in stop_words and len(word) > 2]
        return filtered_word_list
    return []

def load_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                tweet_text = row[2]
                label = row[-1]
                features = preprocess(tweet_text)
                data.append((features, label))
            else:
                print(f"Skipping invalid row: {row}")

    return data


train_data_path = "train_data.csv"
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))


test_string = "i don't love playing football"

print(nb_classifier.classify(preprocess(test_string)))


def accuracy(classifier):
    eval_data_path = "eval_data.csv"
    eval_data = load_data(eval_data_path)

    cnt = 0
    total = len(eval_data)

    for features, true_label in eval_data:
        predicted = classifier.classify(features)
        if predicted == true_label:
            cnt += 1

    accuracy = cnt / total
    return accuracy

accuracy_on_eval_data = accuracy(nb_classifier)

print(f"Accuracy on eval: {accuracy_on_eval_data:.2%}")