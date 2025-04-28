# Importing libraries
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

# Load the dataset
excel = pd.read_excel("/Users/denysenko/Desktop/refined_database.xlsx", index_col=0)

print("Data loaded successfully!")
print(excel.head())

# Preprocessing
# Replace 'X' with 1 and missing values with 0
excel = excel.replace("X", 1)
excel = excel.fillna(0)
excel = excel.apply(pd.to_numeric, errors='coerce')
excel = excel.fillna(0)

print("Data after replacing 'X' and missing values:")
print(excel.head())

# Make sure the data is binary (only 0 and 1)
binary_data = (excel.values > 0).astype(int)

# Computing distances and applying linkage
distances = pairwise_distances(binary_data, metric='euclidean')
print("Pairwise distances computed!")

# Because linkage expects condensed distance matrix
distances_condensed = squareform(distances, checks=False)
linked = linkage(distances_condensed, method='ward')

# Plotting dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked, labels=excel.index, orientation='top', distance_sort='ascending', show_leaf_counts=True)
plt.title('Dendrogram of Papers')
plt.xlabel('Papers')
plt.ylabel('Euclidean Distance')
plt.axhline(y=8.3, color='red', linestyle='--', label='Suggested Cut at 8.3')
plt.legend()
plt.show()


# Choose clusters based on distance
def get_clusters(linked_matrix, distance_cutoff):
    clusters = fcluster(linked_matrix, t=distance_cutoff, criterion='distance')
    return clusters


# Function to count combinations
def find_top_combinations(df, cols, top_n=5):
    all_combinations = Counter()
    for row in df[cols].itertuples(index=False):
        active = [col for col, val in zip(cols, row) if val > 0]
        for size in [2, 3]:
            combos = combinations(active, size)
            all_combinations.update(combos)
    size2 = [(c, n) for c, n in all_combinations.items() if len(c) == 2]
    size3 = [(c, n) for c, n in all_combinations.items() if len(c) == 3]
    return sorted(size2, key=lambda x: -x[1])[:top_n], sorted(size3, key=lambda x: -x[1])[:top_n]


# Find good distances
def suggest_good_distances(linked_matrix, min_clusters=2, max_clusters=10):
    distances = np.linspace(0.5, linked_matrix[:, 2].max(), 100)
    results = []
    seen = set()

    for d in distances:
        clusters = fcluster(linked_matrix, t=d, criterion='distance')
        num = len(set(clusters))
        if min_clusters <= num <= max_clusters:
            sizes = pd.Series(clusters).value_counts()
            config = (num, sizes.max(), sizes.min())
            if config not in seen:
                seen.add(config)
                results.append({
                    'Distance': round(d, 2),
                    'NumClusters': num,
                    'MaxSize': sizes.max(),
                    'MinSize': sizes.min(),
                    'SizeGap': sizes.max() - sizes.min()
                })
    return results


# Print suggested distances
recommendations = suggest_good_distances(linked)
print("Suggested distances for clustering:")
for rec in recommendations:
    print(
        f"Distance: {rec['Distance']}, Clusters: {rec['NumClusters']}, Max Size: {rec['MaxSize']}, Min Size: {rec['MinSize']}, Gap: {rec['SizeGap']}")

# Let user input a distance
chosen_distance = float(input("Enter the distance to cut the dendrogram at: "))

# Assign clusters
cluster_labels = get_clusters(linked, chosen_distance)
excel['Cluster'] = cluster_labels

# Analyze each cluster
print("\nAnalyzing clusters...\n")
columns = excel.columns[:-1]

for cluster_id in np.unique(cluster_labels):
    subset = excel[excel['Cluster'] == cluster_id]
    cluster_size = len(subset)

    print("=" * 60)
    print(f"Cluster {cluster_id} | Size: {cluster_size}")
    print("-" * 60)

    summary = subset[columns].sum()
    percents = summary / cluster_size * 100

    print("Attribute Presence:")
    for attr, val in summary.items():
        print(f"  - {attr}: {int(val)} papers ({percents[attr]:.1f}%)")

    size2, size3 = find_top_combinations(subset, columns)

    print("\nTop 2-Attribute Combinations:")
    for combo, count in size2:
        print(f"  - {', '.join(combo)}: {count} times")

    print("\nTop 3-Attribute Combinations:")
    for combo, count in size3:
        print(f"  - {', '.join(combo)}: {count} times")

    print("=" * 60)
