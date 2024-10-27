import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# Load the data
data = {
    "VIN": ["1N4AZ0CP5D", "1N4AZ1CP8K", "5YJXCAE28L", "SADHC2S1XK", "JN1AZ0CP9B", "1G1RB6S58J", 
            "5YJ3E1EB7K", "3FA6P0SU5E", "5YJ3E1EB3K", "1C4JJXP6XN", "5YJSA1E29L", "5YJYGDEE3L", 
            "JHMZC5F1XJ", "1N4AZ0CP1D", "1N4AZ1BP4L", "KMHC75LH5K", "5YJ3E1EBXJ", "5YJ3E1EA5K", 
            "WA1F2AFY8N"],
    "Electric Range": [75, 150, 293, 234, 73, 53, 220, 19, 220, 21, 330, 291, 47, 75, 149, 29, 215, 220, 23],
    "Base MSRP": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

# Preprocess the data
X = df[['Electric Range', 'Base MSRP']]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal number of clusters
inertia = []
silhouette_scores = []
K = range(2, 10)

for k in K:
    kmedoids = KMeans(n_clusters=k, random_state=42)
    kmedoids.fit(X_scaled)
    inertia.append(kmedoids.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmedoids.labels_))

# Plot the Elbow Method
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

# Run K-Medoids with the optimal k (let's use k=3 for this example)
k = 3  # Number of clusters
kmedoids = KMeans(n_clusters=k, random_state=42)
y_kmedoids = kmedoids.fit_predict(X_scaled)

# Visualization of Clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmedoids, cmap='viridis', marker='o', label='Data Points')
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Medoids Clustering')
plt.xlabel('Electric Range (scaled)')
plt.ylabel('Base MSRP (scaled)')
plt.legend()
plt.grid()
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(df[['Electric Range', 'Base MSRP']])
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()

# Cluster Distribution
df['Cluster'] = y_kmedoids
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Cluster', palette='viridis')
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Vehicles')
plt.show()
