from numpy.random import choice
from numpy import inf, sqrt, array, mean, append


def euclid_distance(a, b):
    return sqrt(sum((a - b) ** 2))

def kmeans(data, n_clusters=2, max_iter=50):
    rows, cols = data.shape
    means = data[choice(range(rows), n_clusters)]
    costs, clusters = [], []
    for _ in range(max_iter):
        clusters, distances = array([]), []
        for i in range(rows):
            min_ind, min_val = -1, inf
            for j in range(n_clusters):
                dis = euclid_distance(data[i], means[j])
                if dis < min_val:
                    min_ind, min_val = j, dis
            clusters = append(clusters, [min_ind], axis=0)
        for i in range(n_clusters):
            means[i] = mean(data[clusters == i], axis=0)
        costs.append(sum(distances))
    return clusters, costs, means


from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = load_iris().data
clusters, costs, means = kmeans(data, 3, 100)
print(silhouette_score(data, clusters))

model = KMeans(3)
model.fit(data)
print(silhouette_score(data, model.labels_))


from sklearn.decomposition import PCA
data = PCA(n_components=2).fit_transform(data)
colors = array(['red', 'green', 'blue'])
plt.scatter(data[:, 0], data[:, 1], c=colors[load_iris().target])
plt.show()

plt.title("Elbow method to find the best k")
plt.xlabel("k")
plt.ylabel("Cost")
costs = []
for n_clusters in range(2, 10):
    model = KMeans(n_clusters). fit(data)
    costs.append(model.inertia_)
plt.plot(range(2, 10), costs)
plt.show()
