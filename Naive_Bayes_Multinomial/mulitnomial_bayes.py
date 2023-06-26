from collections import defaultdict
from math import log

from typing import Dict, List, Tuple

class MultiNomialBayes:
    """"
    Base Class for Multinomial Bayes Classifier

    Args: 
        articles_per_class [Dict]: A map containing the class [str] as keys,
                                with their corresponding articles [List[str]]
                                as values
        laplace_smoothing [Tuple]: The Laplace smoothing parameter for 0 values
        laplace_smoothing[0] -> adds to nominator of the likelihoods
        laplace_smoothing[1] -> adds to denominator of the likelihoods

    """
    def __init__(
        self, articles_per_class: Dict[str, List[str]], 
        laplace_smoothing: Tuple[int, int] = (1,2), 
        missing_word_probability: float = 0.5
    ):
        self.articles_per_class = articles_per_class
        self.classes = list(self.articles_per_class.keys())
        self.likelihoods_per_word_per_class =  {}
        self.laplace_smoothing = laplace_smoothing
        self.missing_word_probability = missing_word_probability
        self.train()


    def train(self):
        """
        calculates different values that are needed to predict the class probabilites for some text

        class_counts_map = number of articles per class 
        priors_per_class = prior probability for a class
        likelihoods_per_word_per_class = the probability for a given word belonging to a given class
        """
        class_counts_map = {len(self.articles_per_class[c]) for c in self.classes}
        self.priors_per_class = {c: number_articles / sum(class_counts_map.values()) \
                                    for c, number_articles in class_counts_map.items()}
        self.likelihoods_per_word_per_class = self.__get_likelihoods_per_word_per_class()


    def predict(self, article: List[str]) -> List[float]:
        """
        predicts the class label probabilites for a given input article / text

        Args:
            article [List]: a list of strings, for which the class should be predicted

        Returns:
            posteriors_per_class [List]: for each class returns the probability that the given article belongs
        """
        posteriors_per_class = {c: log(prior) for c, prior in self.priors_per_class.items()}
        for word in article:
            for c in self.classes:
                posteriors_per_class[c] = posteriors_per_class[c] + log(self.likelihoods_per_word_per_class[word][c])
        
        return posteriors_per_class


    def __get_likelihoods_per_word_per_class(self) -> Dict[str, Dict[str, float]]:
        """
        calculates the likelihoods per word and per class
        Dict[word][class] = p

        Returns:
            likelihoods [Dict]: a map containing the likelihoods for a given word 
                                and the respective class
        """
        word_frequencies_per_class = defaultdict(lambda: {c: 0 for c in self.classes})
        total_words_per_class = defaultdict(int)
        for c, articles in self.articles_per_class:
            for word in articles:
                word_frequencies_per_class[word][c] += 1
                total_words_per_class[c] += 1

        word_likelihoods_per_class = defaultdict(lambda: {c: self.missing_word_probability for c in self.classes})
        for word, frequency_per_class in word_frequencies_per_class.items():
            for c in self.classes:
                word_likelihoods_per_class[word][c] = \
                (word_frequencies_per_class[word][c] + self.laplace_smoothing[0]) / \
                    (total_words_per_class[c] + self.laplace_smoothing[1])

        return word_likelihoods_per_class
