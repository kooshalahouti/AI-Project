from template import NaiveBayesClassifier
import csv
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def preprocess(tweet_string):
    # clean the data and tokenize it
    # You may need to install the 'nltk' library for tokenization
    stop_words = set(stopwords.words('english'))

    # Remove special characters, URLs, and split into words
    tweet_string = re.sub(r'http\S+|www\S+|https\S+', '',
                          tweet_string, flags=re.MULTILINE)
    tweet_string = re.sub(r'\@\w+|\#', '', tweet_string)
    words = word_tokenize(tweet_string)

    # Remove stopwords and non-alphabetic words
    features = [word.lower() for word in words if word.isalpha()
                and word.lower() not in stop_words]

    return features


def load_data(data_path):
    # load the csv file and return the data
    data = []

    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            # assuming the tweet is in the first column
            features = preprocess(row[0])
            label = row[1]  # assuming the label is in the second column
            data.append((features, label))

    return data


# Set the path to your training data CSV file
train_data_path = "path/to/your/train_data.csv"
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))

test_string = "I love playing football"

print(nb_classifier.classify(preprocess(test_string)))
