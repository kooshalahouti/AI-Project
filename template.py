# Naive Bayes 3-class Classifier
# Authors: Baktash Ansari - Sina Zamani

# complete each of the class methods
from collections import defaultdict
import math
import csv


class NaiveBayesClassifier:

    def __init__(self, classes):
        # initialization:
        # inputs: classes(list) --> list of label names
        # class_word_counts --> frequency dictionary for each class
        # class_counts --> number of instances of each class
        # vocab --> all unique words
        self.classes = classes
        self.class_word_counts = {c: defaultdict(int) for c in classes}
        self.class_counts = {c: 0 for c in classes}
        self.vocab = set()

    def train(self, data):
        # training process:
        # inputs: data(list) --> each item of list is a tuple
        # the first index of the tuple is a list of words and the second index is the label(positive, negative, or neutral)

        for features, label in data:
            self.class_counts[label] += 1
            for word in features:
                self.class_word_counts[label][word] += 1
                self.vocab.add(word)
            # Your Code

    def calculate_prior(self):
        # calculate log prior
        # you can add some attributes to this method

        # Your Code
        log_prior = {c: math.log(
            self.class_counts[c] / sum(self.class_counts.values())) for c in self.classes}
        return log_prior

    def calculate_likelihood(self, word, label):
        # calculate likelihhood: P(word | label)
        # return the corresponding value

        smoothing_factor = 1
        word_count = self.class_word_counts[label][word]
        total_words_in_class = sum(self.class_word_counts[label].values())
        likelihood = math.log(word_count + smoothing_factor) / \
            (total_words_in_class + smoothing_factor * len(self.vocab))
        return likelihood

    def classify(self, features):
        # predict the class
        # inputs: features(list) --> words of a tweet
        log_prior = self.calculate_prior()
        log_likelihoods = {c: sum(self.calculate_likelihood(
            word, c) for word in features) for c in self.classes}
        scores = {c: log_prior[c] + log_likelihoods[c] for c in self.classes}
        best_class = max(scores, key=scores.get)
        return best_class


# Good luck :)
