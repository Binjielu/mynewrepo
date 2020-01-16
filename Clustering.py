import numpy as np
import matplotlib.pyplot as plt


class Clustering:

    def DBI(self, x):
        '''
        Davies-Bouldin index
        maximum of the variance between different clusters over their distance
        pick the minimum value
        '''
        y_hat = self.predict(x)
        var = []
        cluster_dist = []
        for i in range(self.k):
            var.append(np.var(x[y_hat == i]))
        #             cluster = x[y_hat==i]
        #             dist = np.sum((cluster - self.centroids[i])**2)
        #             var.append(np.sqrt(dist)/len(cluster))
        #             cluster_dist.append(np.sqrt(np.sum((np.delete(self.centroids, i, axis=0) - self.centroids[i])**2, axis=1)))

        var = np.vstack(var)  # var.shape = 5x1
        #         cluster_dist = np.vstack(cluster_dist) # cluster_dist.shape = 5 x 4

        out = 0
        for i in range(self.k):
            out += np.max((np.delete(var, i, axis=0) + var[i]) /
                          np.sum((np.delete(self.centroids, i, axis=0) - self.centroids[i]) ** 2, axis=1))

        return out / self.k

    def CHI(self, x):
        '''
        Calinski-Harabasz index
        summation of distance between each clusters over summation of variance of each clusters
        pick the maximum value
        '''
        y_hat = self.predict(x)
        out = 0
        dist = 0
        m = np.mean(x, axis=0)
        for i in range(self.k):
            cluster = x[y_hat == i]
            out += len(cluster) * np.sum((m - self.centroids[i]) ** 2)
            dist += np.sum((cluster - self.centroids[i]) ** 2)

        return out * (x.shape[0] - self.k) / (dist * (self.k - 1))

    #         extra_disp, intra_disp = 0., 0.
    #         mean = np.mean(x, axis=0)
    #         for i in range(self.k):
    #             cluster_k = x[y_hat == i]
    #             mean_k = np.mean(cluster_k, axis=0)
    #             extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
    #             intra_disp += np.sum((cluster_k - mean_k) ** 2)

    #         return (1. if intra_disp == 0. else
    #                 extra_disp * (x.shape[0] - self.k) /
    #                 (intra_disp * (self.k - 1.)))

    def Iindex(self, x):
        '''
        I index
        total sum of distance between any point with each cluster over sum of
        suqare error within each cluster multiple maximum distance between
        clusters
        pick the maximum value
        '''
        y_hat = self.predict(x)
        diffs = np.stack([x] * self.k) - self.centroids.reshape(self.k, 1, x.shape[1])
        dists = np.sum(diffs ** 2)
        var = 0
        for i in range(self.k):
            var += np.var(x[y_hat == i]) * len(x[y_hat == i])
        out = []
        for i in range(k):
            out.append(np.max(np.sum((self.centroids - self.centroids[i]) ** 2, axis=1)))

        return (1 / self.k * dists / var * np.max(out)) ** (1 / x.shape[1])

    def D(self, x):
        '''
        Dunn's indices
        pick the maximum value
        '''
        y_hat = self.predict(x)
        dists = []
        dis = []
        for i in range(self.k):
            cluster = x[y_hat == i]
            n = len(cluster)
            dist = []
            for j in range(n):
                dist.append(np.sum((np.delete(x, j, axis=0) - cluster[j]) ** 2, axis=1))
                di.append(np.sum((np.delete(x, cluster, axis=0) - cluster[j]) ** 2, axis=1))
            dists.append(np.max(dist))
            dis.append(np.min(di))

        return np.min(di) / np.max(dists)

    def S(self, x):
        '''
        Silhouette index
        difference between and within the clusters over maximum of the value of this cluster
        pick the maximum value
        '''
        y_hat = self.predict(x)
        a_x = []
        b_x = []
        for i in range(self.k):
            cluster = x[y_hat == i]
            n = len(cluster)
            dist_a = 0
            dist_b = 0
            for j in range(n):
                dist_a += (1 / (n - 1)) * (np.sum((np.delete(cluster, j, axis=0) - cluster[j]) ** 2))
                dist_b += (1 / n) * (np.sum(cluster - cluster[j]) ** 2)
            a_x.append(dist_a)
            b_x.append(dist_b)
        dif = 0
        for i in range(self.k):
            b_x[i] = np.min(np.delete(b_x, i, axis=0))
            dif += (b_x[i] - a_x[i]) / np.max((b_x[i], a_x[i])) / len(x[y_hat == i])

        return 1 / self.k * dif

    def Silhouette(self, x):
        y_hat = self.predict(x)

        b = np.zeros(y_hat.shape)
        a = np.zeros(y_hat.shape)

        for k in range(self.k):
            cluster = y_hat == k

            dist = DistAll(x[cluster])

            #             b[cluster] = np.sum((x[cluster] - self.centroids[k])**2) / np.sum(y_hat == k)
            b[cluster] = np.mean(dist, axis=1)

            for j in range(self.k):
                if j != k:
                    a[cluster] = np.sum((x[cluster] - self.centroids[j]) ** 2, axis=1)
        a /= self.k - 1

        return (a - b) / np.max(np.vstack([b, a]), axis=0)

    def SilhouettePlot(self, x):
        sil = self.Silhouette(x)
        y_hat = self.predict(x)
        out = []
        sil_out = []
        y_hat_out = []
        for i in range(self.centroids.shape[0]):
            sil_cluster = sil[y_hat == i]
            ind = np.argsort(sil_cluster)[::-1]
            y_hat_cluster = y_hat[y_hat == i]

            sil_out.append(sil_cluster[ind])
            y_hat_out.append(y_hat_cluster[ind])

        sil_out = np.hstack(sil_out)
        y_hat_out = np.hstack(y_hat_out)
        plt.scatter(np.arange(sil_out.shape[0]), sil_out, c=y_hat_out, cmap='jet')
        plt.title(' K = {}'.format(self.k))
        plt.show()

