# Importing necessary libraries for data processing, clustering, and visualization
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

# Loading the refined database.
# Each row is a paper, and each column is an attribute related to Data Literacy teaching strategies.
excel = pd.read_excel("/Users/denysenko/Desktop/refined_database.xlsx", index_col=0)

# Preprocessing: Replacing 'X' (marked attributes) with 1 and missing entries with 0
excel.replace({"X": 1, "": 0}, inplace=True)
excel = excel.apply(pd.to_numeric, errors='coerce').fillna(0)

# Binarizing the dataset to ensure that all features are strictly 0 or 1
data = (excel.values > 0).astype(int)

# Computing the pairwise distances between papers and applying Ward's linkage for hierarchical clustering
linked = linkage(squareform(pairwise_distances(data, metric='euclidean'), checks=False), method='ward')

# Plotting a dendrogram to visualize how papers are grouped into clusters at various distances
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=excel.index, orientation='top', distance_sort='ascending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')

# Drawing a horizontal line to suggest an initial distance cut-off for clustering.
# 8.3 was initially used in my paper, this number is fixed in the code for simplicity.
plt.axhline(y=8.3, color='r', linestyle='--', label='Cut at distance = 8.3')
plt.legend()

plt.show()

# Function to create cluster labels based on a chosen distance threshold
def calculate_clusters(linked, distance_level):
    return fcluster(linked, t=distance_level, criterion='distance')

# Function to find and rank the most common attribute combinations inside clusters
def get_top_combinations(cluster_data, columns, top_n=5):
    def count_combinations(size):
        counter = Counter()
        for row in cluster_data[columns].itertuples(index=False):
            active_cols = [col for col, val in zip(columns, row) if val > 0]
            counter.update(combinations(active_cols, size))
        return counter.most_common(top_n)

    return {"size2": count_combinations(2), "size3": count_combinations(3)}

# Helper function to evaluate how many clusters and their sizes exist at a given distance
def evaluate_clusters(linked, distance):
    clusters = fcluster(linked, t=distance, criterion='distance')
    return clusters, pd.Series(clusters).value_counts()

# Function that recommends good distance levels for clustering, based on number and size balance of clusters
def recommend_distances(linked, min_clusters=2, max_clusters=10, step=0.1):
    max_distance = linked[:, 2].max()
    recommendations, seen_configs = [], set()

    for distance in np.arange(0, max_distance, step):
        clusters, cluster_sizes = evaluate_clusters(linked, distance)
        num_clusters = len(cluster_sizes)

        if min_clusters <= num_clusters <= max_clusters:
            config = (num_clusters, cluster_sizes.max(), cluster_sizes.min())
            if config not in seen_configs:
                seen_configs.add(config)
                recommendations.append({
                    'Distance': round(distance, 2),
                    'Num Clusters': num_clusters,
                    'Max Size': cluster_sizes.max(),
                    'Min Size': cluster_sizes.min(),
                    'Uniformity': cluster_sizes.max() - cluster_sizes.min()
                })

    return recommendations

# Getting recommendations for optimal distance values
recommendations = recommend_distances(linked)

print("Recommended distance values for clustering:")
for rec in recommendations:
    print(f"Distance: {rec['Distance']:.2f}, Num Clusters: {rec['Num Clusters']}, "
          f"Max Size: {rec['Max Size']}, Min Size: {rec['Min Size']}, "
          f"Uniformity: {rec['Uniformity']}")

# Allowing the user to select a clustering threshold interactively
chosen_distance_level = float(input("Enter the clustering distance level: "))

# Generating cluster assignments based on the user's chosen distance
clusters = calculate_clusters(linked, chosen_distance_level)
excel['Cluster'] = clusters
columns_to_check = excel.columns[:-1]  # Excluding the newly added 'Cluster' column

# For each cluster, summarizing the attributes and the most common attribute combinations
for cluster_id in np.unique(clusters):
    cluster_data = excel[excel['Cluster'] == cluster_id]
    cluster_size = len(cluster_data)

    attribute_summary = cluster_data[columns_to_check].sum()
    attribute_percentage = (attribute_summary / cluster_size * 100).to_dict()
    top_combinations = get_top_combinations(cluster_data, columns_to_check)

    print(f"\n{'=' * 50}\nCluster ID: {cluster_id} | Size: {cluster_size}\n{'-' * 50}")
    print("Attribute Summary:")
    for attr, count in attribute_summary.items():
        print(f"  - {attr}: {count} occurrences ({attribute_percentage[attr]:.2f}%)")

    print("\nTop Combinations (Size 2):")
    for combo, count in top_combinations["size2"]:
        print(f"  - {', '.join(combo)}: {count} occurrences")

    print("\nTop Combinations (Size 3):")
    for combo, count in top_combinations["size3"]:
        print(f"  - {', '.join(combo)}: {count} occurrences")
    print(f"{'=' * 50}\n")