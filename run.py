from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import csv
from template import NaiveBayesClassifier
import nltk
nltk.download('punkt')
nltk.download('stopwords')


def preprocess(tweet_string):
    tweet_string = tweet_string.lower()
    tweet_string = re.sub(r'[^\w\s]', '', tweet_string)
    features = word_tokenize(tweet_string)
    stop_words = set(stopwords.words('english'))
    features = [word for word in features if word not in stop_words]

    return features


def load_data(data_path):
    data = []

    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            tweet, label = row[0], row[1]
            data.append((preprocess(tweet), label))

    return data


train_data_path = "train_data.csv"
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))

test_string = "I love playing football"

print(nb_classifier.classify(preprocess(test_string)))
