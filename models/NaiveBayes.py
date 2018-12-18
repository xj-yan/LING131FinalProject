from abc import ABCMeta, abstractmethod
from AbstractClassifier import AbstractClassifier 
from math import log
from collections import defaultdict

class NaiveBayes(AbstractClassifier):
    """
    docstring for NaiveBayes
    

    """
    # override
    def labels(self):
        """
        :return a list of labels
        :return type list
        """
        return self._labels

    def most_possible_label(self, prob_by_label):
        """
        :param prob_by_label: ``{label: probability}``
        :return: most possible label name
        """

        maxProb = 0
        result = ''
        for label in prob_by_label:
            if prob_by_label[label] > maxProb:
                maxProb = prob_by_label[label]
                result = label
        print(prob_by_label)
        return result

    def bayes_rule(self, feature_set):
        """
        Apply the bayes rule to calculate P(feature_value|label, feature_name)s
        :param feature_set: a dictionary of features
        :return: most possible label
        """
        prob_by_label = defaultdict(float)

        for label in self._labels:
            prob_by_label[label] = float(self._label_prob_dist[label])

        for label in self._labels:
            for feature_name, feature_value in feature_set.items():
                # only existing features are counted
                if (label, feature_name) in self._feature_prob_dist.keys():
                    if feature_value in self._feature_prob_dist[label, feature_name].keys():
                        prob_by_label[label] *= float(self._feature_prob_dist[label, feature_name][feature_value])
        total = 0
        for label in prob_by_label.keys():
            total += prob_by_label[label]
        for label in prob_by_label:
            prob_by_label[label] = prob_by_label[label] / total
        # print(prob_by_label)
        return self.most_possible_label(prob_by_label)
    
    # override
    def classify(self, feature_set):
        """
        :param feature_set: a dictionary of features
        :return: most possible label
        """
        return self.bayes_rule(feature_set)

    def label_feature_prob_dist(self, labeled_feature_set):
        """
        :param labeled_feature_set: A list of labeled featuresets, 
            in the form of a list of tuples ``(featureset, label),...``.
        """
        # Frequencies
        label_freq_dist = defaultdict(int)
        feature_freq_dist = defaultdict(lambda: defaultdict(int))
        # feature_values = defaultdict(set)
        feature_names = set()

        for feature_set, label in labeled_feature_set:
        	# Record label frequency distribution
            label_freq_dist[label] += 1
            for feature_name, feature_value in feature_set.items():
                # Record a list of feature names
                feature_names.add(feature_name)
                # Record 'feature_name' can take this value
                # feature_values[feature_name].add(feature_value)
                # Record the frequency of 'feature_value' under 'feature_name' given 'label'
                # problem: 
                # 1. a feature set that does not have this feature
                # 2. 
                feature_freq_dist[label, feature_name][feature_value] += 1
        # print(feature_freq_dist)
        # Probabilities
        label_prob_dist = defaultdict(float)
        feature_prob_dist = defaultdict(lambda: defaultdict(float))

        # P(label) distribution
        label_length = len(labeled_feature_set)
        for label in self._labels:
            label_prob_dist[label] = float(label_freq_dist[label]) / float(label_length)

        # P(feature_value|label, feature_name) distribution
        for ((label, feature_name), freq_dist) in feature_freq_dist.items():
            length = label_freq_dist[label]
            probdist = {}
            for feature_value, value_freq in freq_dist.items():
                # smoothing
                if value_freq == 0:
                    value_freq += 1
                probdist[feature_value] = float(value_freq) / float(length)
            feature_prob_dist[label, feature_name] = probdist
        # print(feature_prob_dist)
        return label_prob_dist, feature_prob_dist

    # override
    def train(self, labeled_feature_set):
        """
        :param labeled_feature_set: A list of labeled featuresets, 
            in the form of a list of tuples ``(featureset, label),...``.
        """
        self._labels = list(set(lf[1] for lf in labeled_feature_set))
        self._label_prob_dist, self._feature_prob_dist = self.label_feature_prob_dist(labeled_feature_set)
        return self

    def __init__(self, labeled_feature_set=None):
        """
        Intitialization
        _labels: labels
           type: list
        _label_prob_dist: P(lable) distribution 
           type: dictionary
        _feature_prob_dist: P(feature_value|label, feature_name) distribution
           type: dictioary of dictionaries
        """
        self._labels = []
        self._label_prob_dist = {}
        self._feature_prob_dist = defaultdict(lambda: defaultdict(float))
        if(labeled_feature_set is not None):
            train(self, labeled_feature_set)


