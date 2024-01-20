from collections import defaultdict
import math


class NaiveBayesClassifier:

    def __init__(self, classes):
        self.classes = classes
        self.class_word_counts = {label: {} for label in classes}
        self.class_counts = {label: 0 for label in classes}
        self.vocab = set()

    def train(self, data):
        for features, label in data:
            if label not in self.classes:
                self.classes.append(label)
                self.class_word_counts[label] = {}
                self.class_counts[label] = 0
            self.class_counts[label] += 1
            for word in features:
                if word not in self.class_word_counts[label]:

                    self.class_word_counts[label][word] = 1
                else:
                    self.class_word_counts[label][word] += 1
                self.vocab.add(word)

    def calculate_prior(self):
        total_instances = sum(self.class_counts.values())
        log_prior = {}

        for label, count in self.class_counts.items():
            if count > 0 and total_instances > 0:
                log_prior[label] = math.log(count / total_instances)
            else:
                log_prior[label] = float('-inf')

        return log_prior

    def calculate_likelihood(self, word, label):
        smoothing_factor = 1
        word_count = self.class_word_counts[label].get(
            word, 0) + smoothing_factor
        total_words = sum(self.class_word_counts[label].values(
        )) + smoothing_factor * len(self.vocab)
        likelihood = math.log(word_count / total_words)
        return likelihood

    def classify(self, features):
        log_prior = self.calculate_prior()
        best_class = None
        max_posterior = float('-inf')

        for label in self.classes:
            posterior = log_prior[label]
            for word in features:
                posterior += self.calculate_likelihood(word, label)

            if posterior > max_posterior:
                max_posterior = posterior
                best_class = label

        return best_class
