import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

from SimpleML.Clustering import Clustering


class GaussianMixtureModel:
    def __init__(self, k):
        self.k = k

    def fit(self, x, iteration=4):
        n = x.shape
        self.mean = np.random.rand(self.k, n[1])
        self.covariance = np.zeros((self.k, n[1], n[1]))
        self.priors = np.ones(self.k) * 1 / self.k
        self.priors[-1] = 1 - np.sum(self.priors[:-1])

        for i in range(self.k):
            a = np.identity(n[1], dtype=np.float64)
            self.covariance[i] = a

        for i in range(1, iteration):
            p = np.zeros((n[0], self.k))
            for k in range(self.k):
                print('{} mean is {}.'.format(k, self.mean[k]))
                print('{} covariance is {}.'.format(k, self.covariance[k]))
                # posterior for between every x point corresponding to each cluster
                p[:, k] = mvn.logpdf(x, self.mean[k], self.covariance[k], allow_singular=True) + np.log(self.priors[k])

            y_hat = p.argmax(axis=1)
            #         print(y_hat)

            for k in range(self.k):
                x_k = x[y_hat == k]  # get the all the x values corresponding to each cluster
                # calculate mean and variance
                self.mean[k] = x_k.mean(axis=0)
                self.covariance[k] = np.cov(x_k.T)
                self.priors[k] = len(x_k) / len(x)  # the ratio of y_k over total observations
        #                 print(self.mean[k])
        #                 print(self.covariance[k])

        return self.mean

    def predict(self, x):
        p = np.zeros((len(x), self.k))
        for k in range(self.k):
            # posterior for between every x point corresponding to each cluster
            p[:, k] = mvn.logpdf(x, self.mean[k], self.covariance[k], allow_singular=True) + np.log(self.priors[k])

        # find the index (label) of max possibility of the posterior / axis = 1 along the row
        return p.argmax(axis=1)


class GMM(Clustering):
    def __init__(self, k):
        self.k = k

    def fit(self, x, iteration=10):
        self.centroids = x[np.random.choice(np.arange(x.shape[0]), size=self.k)]
        self.covs = [np.eye(x.shape[1])] * self.k

        for i in range(iteration):
            y_hat = self.predict(x)

            old_centroids = self.centroids.copy()
            for k in range(self.k):
                self.centroids[k, :] = np.mean(x[y_hat == k], axis=0)
                self.covs[k] = np.cov(x[y_hat == k].T)

            if np.sum((old_centroids - self.centroids) ** 2) < 1e-15:
                break

    def predict(self, x):
        probs = self.probability(x)
        y_hat = np.argmax(probs, axis=0)
        return y_hat

    def probability(self, x):
        probs = []
        for k in range(self.k):
            probs.append(mvn.pdf(x, mean=self.centroids[k], cov=self.covs[k], allow_singular=True))
        probs = np.vstack(probs)
        return probs

    def loss(self, x):
        probs = self.probability(x)
        y_hat = self.predict(x)
        unique, counts = np.unique(y_hat, return_counts=True)
        pis = counts.reshape(-1, 1) / y_hat.shape[0]

        return -np.sum(np.log(np.sum(probs * pis, axis=0) + 1e-99))