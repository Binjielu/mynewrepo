import numpy as np
from Cython.Shadow import NULL
from scipy.stats import multivariate_normal as mvn


class NaiveBayesClassifier:
    def __init__(self):
        self.K = NULL    # number of features
        self.likelihoods = dict()    # p(y|x)
        self.priors = dict()      # p(y)

    def fit(self, x, y):
        self.K = set(y)    # get the unique value in y

        for k in self.K:
            x_k = x[y == k]     # get the all the x values corresponding to specific y value
            # calculate mean and variance
            self.likelihoods[k] = {"mu": x_k.mean(axis=0), "SIGMA": np.cov(x_k.T)}
            self.priors[k] = len(x_k) / len(x)     # the ratio of y_k over total observations

    def predict(self, x):
        p = np.zeros((len(x), len(self.K)))
        # k is the features
        for k, l in self.likelihoods.items():
            # posterior for each features
            p[:, k] = mvn.logpdf(x, l["mu"], l["SIGMA"], allow_singular=True) + np.log(self.priors[k])

        # find the index (label) of max possibility of the posterior / axis = 1 along the row
        return p.argmax(axis=1)

