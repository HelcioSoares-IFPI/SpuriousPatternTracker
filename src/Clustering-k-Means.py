import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.KMeans import *

# Instructions for preparing the input files:
# Copy the 'ErrorsInfluence' files (for classes 0 and 1) and rename them according to the model name and dataset.
# Example:
#     - Model: LRG
#     - Dataset: Contracts
#     - Class: 0
# The file name should follow the pattern: LRG-contracts-0.csv

# Main program to cluster confounders
# This script will call functions from the KMeans library.

# Model, dataset, and class information
model = ''
dataset = ''
class_label = 0

# Define the data file path
data_path = f'{model}-{dataset}-{class_label}'

# Load the dataset
data = pd.read_csv(f'./results/{data_path}.csv')

# Normalize the numeric columns
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data[numeric_columns])

# Calculate the WCSS for normalized data
wcss_normalized = calculate_wcss(data_normalized)

# Determine the optimal number of clusters
ideal_clusters = determine_optimal_clusters(wcss_normalized)

# Plot the WCSS curve and highlight the optimal number of clusters
plot_hm(wcss_normalized, ideal_clusters, model, dataset, class_label, tipo='CO')

# Apply PCA for dimensionality reduction
pca_results, loadings = apply_pca(data_normalized)

# Generate a heatmap for PCA loadings
plot_heatmap(loadings, numeric_columns, dicHeatMapsColuns, model, dataset, class_label, tipo='HM')

# Perform clustering using KMeans on PCA results
kmeans_pca = KMeans(n_clusters=ideal_clusters, random_state=42).fit(pca_results)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2', 'PC3'])
pca_df['Cluster'] = kmeans_pca.labels_
pca_df['Pattern'] = data['padrao']

# Get cluster centroids
pca_centroids = kmeans_pca.cluster_centers_

# Plot an interactive 3D visualization with centroids
plot_3D_i(pca_df, pca_centroids, model, dataset, class_label, tipo='3D')

# Plot a static 3D visualization and return selected points
selected_points, selection_types = plot_3D_e(pca_df, pca_centroids, model, dataset, class_label, tipo='3D')

# Get patterns and cluster labels
patterns = data['padrao'].values
clusters = kmeans_pca.labels_

# Extract indices of selected points
selected_points_indices = selected_points.index.tolist()

# Plot a 2D PCA visualization
plot_2d(
    data_normalized, patterns, clusters, pca_centroids,
    'Principal Component Analysis (PCA) - 2D Reduction',
    model, dataset, class_label, '2D', selected_points_indices,
    n_closest=1, n_middle=1, n_furthest=3
)

# Plot a 1D bar chart of distances to centroids
plot_1D(model, dataset, class_label, '1D')
