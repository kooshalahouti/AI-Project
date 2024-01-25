# import re
import csv
from template import NaiveBayesClassifier
import string


def preprocess(tweet_string):
    tweet_string = tweet_string.lower()
    tweet_string = tweet_string.translate(str.maketrans("", "", string.punctuation))
    features = tweet_string.split()
    return features


def load_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
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

test_string = "I don't love playing football"

print(nb_classifier.classify(preprocess(test_string)))
