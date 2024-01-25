from collections import defaultdict
import math


class NaiveBayesClassifier:

    def __init__(self, classes):
        self.classes = classes
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.vocab = set()        

    def train(self, data):
        for features, label in data:
            self.class_counts[label] += 1
            for word in features:
                self.class_word_counts[label][word] += 1
                self.vocab.add(word)

    def calculate_prior(self):
        total_instances = sum(self.class_counts.values())
        log_prior = {label: math.log(count / total_instances) for label, count in self.class_counts.items()}
        # for label in self.classes:
        #     if label not in log_prior:
        #         log_prior[label] = float('-inf')
        return log_prior

    def calculate_likelihood(self, word, label):
        alpha = 1.0
        numerator = self.class_word_counts[label][word] + alpha
        denominator = sum(self.class_word_counts[label].values()) + (alpha * len(self.vocab))
        return math.log(numerator / denominator)

    def classify(self, features):
        log_prior = self.calculate_prior()
        class_scores = {label: log_prior.get(label, 0.0) for label in self.classes}

        for word in features:
            for label in self.classes:
                class_scores[label] += self.calculate_likelihood(word, label)

        best_class = max(class_scores, key=class_scores.get)
        return best_class
