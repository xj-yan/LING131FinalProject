from abc import ABCMeta, abstractmethod
from AbstractClassifier import AbstractClassifier 
from math import log

class NaiveBayes(AbstractClassifier):
    """
    docstring for NaiveBayes

    """
    def __init__(self):
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

    def __init__(self, labeled_feature_set):
    	train(self, labeled_feature_set)

    # override
    def labels(self):
    	"""
        :return a list of labels
        :return type list
        """
        return self._labels

    # override
    def train(self, labeled_feature_set):
        """
        :param feature_set: a dictionary of features
        :return self: the object itself
        """
        self._labels = list(label_prob_dist.keys())
        self._label_prob_dist, self._feature_prob_dist = feature_prob_dist(self, labeled_feature_set)
        return self

    # override
    def classify(self, feature_set):
    	"""
        :param feature_set: a dictionary of features
        :return: most possible label
        """
        return self.bayes_rule(self, feature_set)

    def bayes_rule(self, feature_set):
    	"""
    	Apply the bayes rule to calculate P(feature_value|label, feature_name)s
        :param feature_set: a dictionary of features
        :return: most possible label
        """
        prob_by_label = {}
        for label in self._labels:
            prob_by_label[label] += log(self._label_prob_dist[label])
            for (feature_name, feature_value) in feature_set.items():
                # only existing features are counted
                if (label, feature_name) in self._feature_prob_dist:
                    prob_by_label[label] += log(self._feature_prob_dist[label, feature_name][feature_value])
        return most_possible_label(prob_by_label)

    def most_possible_label(prob_by_label):
        """
        :param prob_by_label: ``{label: probability}``
        :return: most possible label name
        """
        result_label = ''
        max_prob = 0
        for label, prob in prob_by_label.items():
            if prob > max_prob:
                result_label = label
                max_prob = prob
        return result_label

    def label_feature_prob_dist(self, labeled_feature_set):
        """
        :param labeled_feature_set: A list of labeled featuresets, 
            in the form of a list of tuples ``(featureset, label),...``.
        """
        # Frequencies
        label_freq_dist = {}
        feature_freq_dist = defaultdict(lambda: defaultdict(int))
        feature_values = defaultdict(set)
        feature_names = set()

        for feature_set, label in labeled_feature_set:
        	# Record label frequency distribution
            label_freq_dist[label] += 1
            for feature_name, feature_value in feature_set.items():
                # Record a list of feature names
                feature_names.add(feature_name)
                # Record 'feature_name' can take this value
                feature_values[feature_name].add(feature_value)
                # Record the frequency of 'feature_value' under 'feature_name' given 'label'
                feature_freq_dist[label, feature_name][feature_value] += 1

        # Probabilities
        label_prob_dist = {}
        feature_prob_dist = defaultdict(lambda: defaultdict(float))

        # P(label) distribution
        length = len(labeled_feature_set)
        for label in _labels:
            label_prob_dist[label] = label_freq_dist[label] / float(length)

        # P(feature_value|label, feature_name) distribution
        for ((label, feature_name), freq_dist) in feature_freq_dist.items():
            length = len(feature_values[feature_name])
            probdist = {}
            for feature_value, value_freq in freq_dist.items():
                probdist[feature_value] = value_freq / float(length)
            feature_prob_dist[label, feature_name] = probdist

        return label_prob_dist, feature_prob_dist

