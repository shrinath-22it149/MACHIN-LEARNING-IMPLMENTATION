import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample data in a dictionary format (replace this with your actual data source)
data = {
    'VIN': ['1N4AZ0CP5D', '1N4AZ1CP8K', '5YJXCAE28L', 'SADHC2S1XK', 'JN1AZ0CP9B',
            '1G1RB6S58J', '5YJ3E1EB7K', '3FA6P0SU5E', '5YJ3E1EB3K', '1C4JJXP6XN',
            '5YJSA1E29L', '5YJYGDEE3L', 'JHMZC5F1XJ', '1N4AZ0CP1D', '1N4AZ1BP4L',
            'KMHC75LH5K', '5YJ3E1EBXJ', '5YJ3E1EA5K', 'WA1F2AFY8N'],
    'Electric Range': [75, 150, 293, 234, 73, 53, 220, 19, 220, 21,
                      330, 291, 47, 75, 149, 29, 215, 220, 23],
    'Base MSRP': [0] * 19  # Assuming Base MSRP is not required for clustering
}

# Create a DataFrame
df = pd.DataFrame(data)

# Select relevant features for clustering
X = df[['Electric Range']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(df['Electric Range'], [0] * len(df), c=df['Cluster'], cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], [0] * kmeans.n_clusters, c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering of Electric Vehicles')
plt.xlabel('Electric Range (miles)')
plt.yticks([])  # Hide y-axis
plt.legend()
plt.grid()
plt.show()

# Print initial and final clusters
print("Initial Clusters (before fitting):")
print(X)

print("\nFinal Clusters (after fitting):")
print(df[['VIN', 'Electric Range', 'Cluster']])