import numpy as np
from SimpleML.Clustering import Clustering

# def DistAll(x):
#     diff = (x.reshape(1, x.shape[0], x.shape[1]) - x.reshape(x.shape[0], 1, x.shape[1]))**2
#     return np.sum(diff, axis=2)


def var(x):
    return np.sum((x - np.mean(x, axis = 0)) ** 2 ) / x.shape[0]

def DistAll(x):
    diff = (x.reshape(1, x.shape[0], x.shape[1]) - x.reshape(x.shape[0], 1, x.shape[1]))
    return np.sum(diff ** 2, axis=2)

def Responsibility(x, centroid, beta):
    diff = x - centroid
    dist = np.sum(diff ** 2, axis=1)
    return np.exp(-beta * dist) / np.sum(np.exp(-beta * dist))

def Dist(x, centroid):
    diff = x - centroid
    return np.sum(diff ** 2, axis=1)




class Kmeans:
    def __init__(self, k):
        self.k = k

    def fit(self, x, iteration=10):

        self.centroids = x[np.random.choice(np.arange(x.shape[0]), size=self.k)]

        for i in range(iteration):
            #             print('\rIteration: {}'.format(i), end='')
            diffs = np.stack([x] * self.k) - self.centroids.reshape(self.k, 1, x.shape[1])
            dist = np.sum(diffs ** 2, axis=2)
            y_hat = np.argmin(dist, axis=0)

            old_centroids = self.centroids.copy()
            for i in range(self.k):
                self.centroids[i ,:] = np.mean(x[y_hat == i], axis=0)

            if (old_centroids == self.centroids).all():
                #                 print(old_centroids == self.centroids)
                break
        return self.centroids

    def predict(self, x):
        diffs = np.stack([x] * self.k) - self.centroids.reshape(self.k, 1, x.shape[1])
        dist = np.sum(diffs ** 2, axis=2)
        y_hat = np.argmin(dist, axis=0)

        return y_hat


class DBSCAN:
    def __init__(self, epsilon=0.01, neighbor_threshold=20):
        self.epsilon = epsilon
        self.neighbor_threshold = neighbor_threshold

    def fit(self, x):
        dist = DistAll(x)
        neighbor = dist < self.epsilon
        neighbor_count = np.sum(neighbor, axis=0)
        core = neighbor_count > self.neighbor_threshold

        avail = core.copy()
        clusters = []
        while avail.any():
            ind = np.random.choice(np.where(avail)[0])
            cluster = neighbor[ind]
            old_cluster = np.zeros(cluster.shape)

            while np.any(cluster != old_cluster):
                old_cluster = cluster.copy()
                cluster = np.any(cluster | neighbor[cluster & core], axis=0)
            avail = avail & ~cluster
            clusters.append(cluster)

        y_hat = np.vstack(clusters)
        noise = ~np.any(y_hat, axis=0)
        y_hat = np.vstack([noise, y_hat])

        self.core_point = x[core]
        self.core_class = np.argmax(y_hat, axis=0)[core]

    def predict(self, x):
        dist = np.sum((x.reshape(1, -1, x.shape[1]) - self.core_point.reshape(-1, 1, x.shape[1])) ** 2, axis=2)
        neighbor = dist < self.epsilon
        y_hat = self.core_class[np.argmin(dist, axis=0)]
        #         print(np.argmin(dist, axis=0))
        y_hat[~np.any(neighbor, axis=0)] = 0

        return y_hat


class SoftKMeans(Clustering):
    def __init__(self, k, beta):
        self.k = k
        self.beta = beta

    def Fit(self, x, iterations=10):
        # Step 1
        self.centroids = x[np.random.choice(np.arange(x.shape[0]), size=self.k)]
        for i in range(iterations):
            print('\rIterations: {} '.format(i), end='')
            # Step 2
            diffs = np.stack([x] * self.k) - self.centroids.reshape(self.k, 1, x.shape[1])
            dist = np.sum(diffs ** 2, axis=2)
            resp = np.exp(-self.beta * dist) / np.sum(np.exp(-self.beta * dist), axis=1, keepdims=True)
            # Step 3
            y_hat = np.argmax(resp, axis=0)
            # Step 4
            old_centroids = self.centroids.copy()
            for i in range(self.k):
                resp_i = resp[i, y_hat == i].reshape(-1, 1)
                self.centroids[i, :] = np.sum(x[y_hat == i] * resp_i, axis=0) / (np.sum(resp_i) + 1e-99)
            #             print(np.sum((old_centroids - self.centroids) ** 2))
            if (np.sum((old_centroids - self.centroids) ** 2)) < 1e-15:
                break

        return self.centroids

    def Predict(self, x):
        # Step 2
        diffs = np.stack([x] * self.k) - self.centroids.reshape(self.k, 1, x.shape[1])
        dist = np.sum(diffs ** 2, axis=2)
        resp = np.exp(-self.beta * dist) / np.sum(np.exp(-self.beta * dist), axis=1, keepdims=True)
        # Step 3
        y_hat = np.argmax(resp, axis=0)

        return y_hat
