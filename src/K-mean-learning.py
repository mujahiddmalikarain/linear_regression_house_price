from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data: [visits, spend]
data = [[2, 50], [10, 600], [1, 40], [9, 580], [3, 100]]

# Create the model (K=2 clusters)
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)

# Get cluster assignments
labels = kmeans.labels_

# Plot clusters
for i in range(2):
    cluster = [data[j] for j in range(len(data)) if labels[j] == i]
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    plt.scatter(x, y, label=f"Cluster {i+1}")

plt.xlabel("Visit Frequency")
plt.ylabel("Monthly Spend")
plt.legend()
plt.show()
