import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import warnings
import logging
import sys
import os

# Suppress warning messages
warnings.filterwarnings("ignore")

# Suppress logs from libraries like matplotlib
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

# List of numeric columns used in analysis
numeric_columns = [
    'freqTreinoMesmaClasse', 'freqTreinoOutraClasse',
    'freqErroTesteMesmaClasse', 'freqErroTesteOutraClasse',
    'div_frequenciasOM', 'div_freqErroOM',
    'peso_global', 'pIdx', 'pQtd',
    'aIdx', 'aQtd'
]

# Dictionary mapping metrics to Greek symbols for heatmap visualization
dicHeatMapsColuns = {
    'freqTreinoMesmaClasse': r'$\mu_t$', 'freqTreinoOutraClasse': r'$\phi_t$',
    'freqErroTesteMesmaClasse': r'$\mu_{e}$', 'freqErroTesteOutraClasse': r'$\phi_e$',
    'div_frequenciasOM': r'$\kappa_t$', 'div_freqErroOM': r'$\kappa_e$',
    'peso_global': r'$\rho$', 'pIdx': r'$\tau_p$',
    'pQtd': r'$\varrho_p$', 'aIdx': r'$\tau_a$', 'aQtd': r'$\varrho_a$'
}


def rgb_string_to_tuple(rgb_string):
    """
    Convert an RGB string (e.g., 'rgb(255, 0, 0)') into a normalized tuple (0-1 scale).
    """
    rgb_values = rgb_string.replace('rgb(', '').replace(')', '').split(', ')
    return tuple(int(v) / 255 for v in rgb_values)


# Colors for cluster visualization
cluster_colors_ = [
    'rgb(232, 38, 6)', 'rgb(68, 1, 84)', 'rgb(24, 17, 227)',
    'rgb(55, 126, 71)', 'rgb(253, 231, 37)', 'rgb(0, 255, 0)'
]
cluster_colors = [rgb_string_to_tuple(color) for color in cluster_colors_]


# Class to suppress print outputs
class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Function to calculate Within-Cluster Sum of Squares (WCSS)
def calculate_wcss(data):
    """
    Calculate the WCSS for a range of cluster numbers.
    """
    wcss = []
    for n in range(1, 11):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss


# Function to determine the optimal number of clusters
def determine_optimal_clusters(wcss, n_points=10):
    """
    Determine the optimal number of clusters using the curvature method.
    """
    curvatures = []
    for i in range(1, n_points - 1):
        prev = wcss[i - 1]
        current = wcss[i]
        next = wcss[i + 1]
        curvature = abs((next - 2 * current + prev))
        curvatures.append(curvature)

    max_curvature_index = np.argmax(curvatures)
    optimal_k = max_curvature_index + 2
    return optimal_k


# Function to plot WCSS and highlight the optimal number of clusters
def plot_hm(wcss, optimal_k, model, dataset, class_label, data_type):
    """
    Plot WCSS and highlight the optimal number of clusters.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, 11), wcss, marker='o', label='WCSS', color='blue')
    ax.plot(optimal_k, wcss[optimal_k - 1], marker='o', markersize=10,
            label=f'Optimal number of clusters: {optimal_k}', color='green', zorder=2)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    ax.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'./results/{model}-{data_type}-{dataset}-{class_label}.png', dpi=600)
    plt.show()


# Function to generate heatmap with labels from the dictionary
def plot_heatmap(data, columns, dicHeatMapsColuns, model, dataset, class_label, data_type):
    """
    Generate a heatmap using PCA loadings and save the visualization and data.
    """
    greek_labels = [dicHeatMapsColuns[col] if col in dicHeatMapsColuns else col for col in columns]
    loadings_df = pd.DataFrame(data.T, columns=['PC1', 'PC2', 'PC3'], index=greek_labels)
    loadings_df = loadings_df.round(2)

    plt.figure(figsize=(12, 6))
    sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Weight'})
    plt.ylabel('Metrics')
    plt.tight_layout()

    image_name = f'./results/{model}-{data_type}-{dataset}-{class_label}.png'
    plt.savefig(image_name, dpi=600)
    plt.show()

    csv_name = f'./results/{model}-{data_type}-{dataset}-{class_label}.csv'
    loadings_df.to_csv(csv_name, index_label='Metrics')



# Function to calculate the distances between data points and centroids
def calculate_distances(data, centroids):
    """
    Calculates the Euclidean distance of each point from its assigned cluster centroid.

    Parameters:
    - data (DataFrame): Input data with cluster assignments.
    - centroids (list): List of centroid coordinates for each cluster.

    Returns:
    - DataFrame: Data with an additional column 'Distance_to_Centroid'.
    """
    data_with_distances = data.copy()
    distances = []
    for i, row in data.iterrows():
        centroid = centroids[int(row['Cluster'])]
        distance = np.sqrt((row['PC1'] - centroid[0])**2 + (row['PC2'] - centroid[1])**2 + (row['PC3'] - centroid[2])**2)
        distances.append(distance)
    data_with_distances['Distance_to_Centroid'] = distances
    return data_with_distances

# Function to apply PCA for dimensionality reduction
def apply_pca(data, n_components=3):
    """
    Applies PCA (Principal Component Analysis) to reduce the dimensionality of the data.

    Parameters:
    - data (DataFrame or array): Input data.
    - n_components (int): Number of principal components to retain.

    Returns:
    - tuple: (principal components, PCA loadings).
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    loadings = pca.components_
    return principal_components, loadings

# Function to select 'n' closest, middle, and furthest points based on distances
def select_n_points_2d(distances, n_closest=1, n_middle=1, n_furthest=3):
    """
    Selects the indices of the closest, middle, and furthest points based on distances.

    Parameters:
    - distances (array): Array of distances for points in the data.
    - n_closest (int): Number of closest points to select.
    - n_middle (int): Number of middle-distance points to select.
    - n_furthest (int): Number of furthest points to select.

    Returns:
    - array: Concatenated indices of selected points.
    """
    indices = np.argsort(distances)  # Sort indices by distance
    closest_indices = indices[:n_closest]
    furthest_indices = indices[-n_furthest:]
    middle_start = len(indices) // 2 - (n_middle // 2)
    middle_indices = indices[middle_start: middle_start + n_middle]
    return np.concatenate([closest_indices, middle_indices, furthest_indices])

# Function to select 'n' closest, middle, and furthest points for each cluster
def select_n_points(data_with_distances, n_closest=1, n_middle=1, n_furthest=3):
    """
    Selects 'n' closest, middle, and furthest points for each cluster.

    Parameters:
    - data_with_distances (DataFrame): Data containing distance and cluster assignments.
    - n_closest (int): Number of closest points to select.
    - n_middle (int): Number of middle-distance points to select.
    - n_furthest (int): Number of furthest points to select.

    Returns:
    - DataFrame: Selected points from all clusters.
    - list: Types of selection for each point ('Closest', 'Medium', 'Most_Distant', or 'Not_Selected').
    """
    selected_points = pd.DataFrame()
    selection_types = ['Not_Selected'] * len(data_with_distances)

    for cluster in data_with_distances['Cluster'].unique():
        cluster_data = data_with_distances[data_with_distances['Cluster'] == cluster]
        sorted_data = cluster_data.sort_values(by='Distance_to_Centroid')

        # Select closest points
        closest_points = sorted_data.iloc[0:n_closest]

        # Select middle-distance points
        middle_start = len(sorted_data) // 2 - (n_middle // 2)
        middle_points = sorted_data.iloc[middle_start: middle_start + n_middle]

        # Select furthest points
        furthest_points = sorted_data.iloc[-n_furthest:]

        # Mark selected points
        for idx in closest_points.index:
            selection_types[idx] = 'Closest'
        for idx in middle_points.index:
            selection_types[idx] = 'Medium'
        for idx in furthest_points.index:
            selection_types[idx] = 'Most_Distant'

        # Concatenate selected points
        selected_points = pd.concat([selected_points, closest_points, middle_points, furthest_points])

    return selected_points, selection_types

# Function to plot a 2D reduction and save the output as CSV and image
def plot_2d(data_normalized, labels, clusters, centroids, title, model, dataset, class_label, data_type, selected_points_indices):
    """
    Reduces data to 2D using PCA and plots the clusters with annotations for selected points.

    Parameters:
    - data_normalized (array): Normalized data.
    - labels (list): Labels for data points.
    - clusters (array): Cluster assignments for each data point.
    - centroids (array): Coordinates of cluster centroids.
    - title (str): Plot title.
    - model (str): Model name.
    - dataset (str): Dataset name.
    - class_label (int): Class label.
    - data_type (str): Type of analysis (e.g., train/test).
    - selected_points_indices (list): Indices of selected points.
    """
    data, _ = apply_pca(data_normalized, 2)

    plt.figure(figsize=(26, 14))
    unique_clusters = np.unique(clusters)

    # Plot points by cluster
    for idx, cluster in enumerate(unique_clusters):
        cluster_data = data[clusters == cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=cluster_colors[idx], alpha=0.5,
                    edgecolor='k', label=f'Cluster {cluster}', zorder=3, s=100)

    # Plot centroids as diamonds
    for i, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], color=cluster_colors[i], marker='D', s=180, edgecolors='black', zorder=4)

    # Annotate selected points
    for idx in selected_points_indices:
        plt.annotate(labels[idx], (data[idx, 0], data[idx, 1]), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=26, color=cluster_colors[clusters[idx]], zorder=5)
        plt.scatter(data[idx, 0], data[idx, 1], color=cluster_colors[clusters[idx]],
                    s=240, edgecolors='black', zorder=6)

    plt.xlabel('PC1', fontsize=24)
    plt.ylabel('PC2', fontsize=24)
    plt.grid(True, linestyle='--', linewidth=0.5, zorder=0)
    plt.tight_layout()

    # Add legend
    handles = [plt.Line2D([0], [0], color=cluster_colors[i], lw=4) for i in range(len(unique_clusters))]
    plt.legend(handles, [f'Cluster {i}' for i in range(len(unique_clusters))], loc='upper center',
               bbox_to_anchor=(0.87, 1), fancybox=True, shadow=True, ncol=len(unique_clusters), fontsize=24)

    # Save image and CSV
    image_name = f'./results/{model}-{data_type}-{dataset}-{class_label}.png'
    plt.savefig(image_name, dpi=600)
    plt.show()

    df = pd.DataFrame(data, columns=['PC1', 'PC2'])
    df['Cluster'] = clusters
    df['Pattern'] = labels
    df['Centroid_PC1'] = [centroids[cluster][0] for cluster in clusters]
    df['Centroid_PC2'] = [centroids[cluster][1] for cluster in clusters]
    csv_name = f'./results/{model}-{data_type}-{dataset}-{class_label}.csv'
    df.to_csv(csv_name, index_label='Index')



# Function to create a 3D interactive plot with centroids and save data to CSV
def plot_3D_i(data, centroids, model, dataset, class_label, data_type, n_closest=1, n_middle=1, n_furthest=3):
    """
    Creates a 3D interactive scatter plot with centroids and highlighted patterns.

    Parameters:
    - data (DataFrame): Data points to visualize.
    - centroids (array): Coordinates of cluster centroids.
    - model (str): Model name.
    - dataset (str): Dataset name.
    - class_label (int): Class label.
    - data_type (str): Type of analysis (e.g., train/test).
    - n_closest (int): Number of closest points to highlight.
    - n_middle (int): Number of middle-distance points to highlight.
    - n_furthest (int): Number of furthest points to highlight.
    """
    # Create a DataFrame for centroids
    centroid_df = pd.DataFrame(centroids, columns=['PC1', 'PC2', 'PC3'])
    centroid_df['Cluster'] = range(len(centroids))
    centroid_df['Pattern'] = ['Centroid'] * len(centroids)
    centroid_df['Distance_to_Centroid'] = [0] * len(centroids)  # Distance from centroid to itself is zero
    centroid_df['Selection_Type'] = ['Centroid'] * len(centroids)  # Mark as centroid

    # Calculate distances of each point to its respective centroid
    data_with_distances = calculate_distances(data, centroids)

    # Select 'n' closest, middle, and furthest points
    selected_points, selection_types = select_n_points(data_with_distances, n_closest, n_middle, n_furthest)

    # Add selection information to the DataFrame
    data_with_distances['Selection_Type'] = selection_types

    # Combine centroid data and normal data
    all_data = pd.concat([data_with_distances, centroid_df], ignore_index=True)

    # Define a custom color palette for clusters
    cluster_colors = ['rgb(232, 38, 6)', 'rgb(68, 1, 84)', 'rgb(24, 17, 227)', 'rgb(55, 126, 71)', 'rgb(253, 231, 37)', 'rgb(0, 255, 0)']

    # Map colors to clusters
    data['color'] = data['Cluster'].apply(lambda x: cluster_colors[x])
    mod = dataset.capitalize()[0]
    title = f'{model}-{mod}{class_label}: 3D visualization of clustering with PCA and centroids to identify spurious patterns.'

    # Create the 3D scatter plot
    fig = px.scatter_3d(
        data, x='PC1', y='PC2', z='PC3',
        labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3', 'Cluster': 'Cluster'},
        title=title, text=None,
        hover_data={'Cluster': False, 'Pattern': True, 'PC1': True, 'PC2': True, 'PC3': True}
    )

    # Update marker colors manually
    fig.update_traces(marker=dict(color=data['color'], size=5))

    # Add centroids to the plot
    for cluster in range(len(centroids)):
        fig.add_trace(go.Scatter3d(
            x=[centroids[cluster][0]], y=[centroids[cluster][1]], z=[centroids[cluster][2]],
            mode='markers+text', marker=dict(size=4, color=cluster_colors[cluster], symbol='diamond'),
            text=[''],
            name=f'Cluster {cluster} Centroid',
            hovertemplate='<b>PC1: %{x:.4f}</b><br>'
                            '<b>PC2: %{y:.4f}</b><br>'
                            '<b>PC3: %{z:.4f}</b><br>'
                            '<b>Type: %{text}</b>'
        ))

    # Add highlighted points without showing them in the legend
    fig.add_trace(go.Scatter3d(
        x=selected_points['PC1'], y=selected_points['PC2'], z=selected_points['PC3'],
        mode='markers+text', marker=dict(size=6, color=selected_points['Cluster'].apply(lambda x: cluster_colors[x]),
                                        symbol='circle', line=dict(color='black', width=2)),
        textfont=dict(size=13),
        text=selected_points['Pattern'],
        textposition='top center',
        name='Highlighted Patterns',
        showlegend=False,
        hovertemplate='<b>PC1: %{x:.4f}</b><br>'
                      '<b>PC2: %{y:.4f}</b><br>'
                      '<b>PC3: %{z:.4f}</b><br>'
                      '<b>Pattern: %{text}</b>'
    ))

    # Adjust the size of unhighlighted points
    fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))

    # Layout improvements
    fig.update_layout(
        showlegend=True,
        legend=dict(
            title="Color Legend",
            itemsizing="constant",
            traceorder="normal",
            font=dict(size=12),
            orientation="h",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.35
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(245, 245, 245, 0.1)',
                zerolinecolor='rgba(0, 0, 0, 0)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.1)',
                zerolinecolor='rgba(0, 0, 0, 0)',
                showspikes=False
            ),
            zaxis=dict(
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.1)',
                zerolinecolor='rgba(0, 0, 0, 0)'
            )
        )
    )

    # Save the interactive plot as an HTML file
    fig.write_html(f'./results/{model}-{data_type}-{dataset}-{class_label}.html')

    # Save the data to a CSV file
    csv_name = f'./results/{model}-{data_type}-{dataset}-{class_label}.csv'
    all_data.to_csv(csv_name, index_label='Index')

# Function to create a static 3D plot with centroids and save as PNG
def plot_3D_e(data, centroids, model, dataset, class_label, data_type, n_closest=1, n_middle=1, n_furthest=3):
    """
    Creates a static 3D plot with centroids and highlighted patterns and saves the result as a PNG.

    Parameters:
    - data (DataFrame): Data points to visualize.
    - centroids (array): Coordinates of cluster centroids.
    - model (str): Model name.
    - dataset (str): Dataset name.
    - class_label (int): Class label.
    - data_type (str): Type of analysis (e.g., train/test).
    - n_closest (int): Number of closest points to highlight.
    - n_middle (int): Number of middle-distance points to highlight.
    - n_furthest (int): Number of furthest points to highlight.

    Returns:
    - DataFrame: Selected points.
    - list: Selection types for each data point.
    """
    # Create a DataFrame for centroids
    centroid_df = pd.DataFrame(centroids, columns=['PC1', 'PC2', 'PC3'])
    centroid_df['Cluster'] = range(len(centroids))
    centroid_df['Pattern'] = ['Centroid'] * len(centroids)
    centroid_df['Distance_to_Centroid'] = [0] * len(centroids)  # Distance from centroid to itself is zero
    centroid_df['Selection_Type'] = ['Centroid'] * len(centroids)  # Mark as centroid

    # Calculate distances of each point to its respective centroid
    data_with_distances = calculate_distances(data, centroids)

    # Select 'n' closest, middle, and furthest points
    selected_points, selection_types = select_n_points(data_with_distances, n_closest, n_middle, n_furthest)

    # Add selection information to the DataFrame
    data_with_distances['Selection_Type'] = selection_types

    # Create a static 3D plot
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all dataset points, colored by cluster, with smaller size for unselected patterns
    for i, cluster in enumerate(np.unique(data['Cluster'])):
        cluster_data = data[data['Cluster'] == cluster]
        ax.scatter(cluster_data['PC1'], cluster_data['PC2'], cluster_data['PC3'],
                   color=cluster_colors[i % len(cluster_colors)], label=f'Cluster {cluster}', s=10)

    # Add centroids to the plot with smaller size
    for i, cluster in enumerate(range(len(centroids))):
        centroid = centroids[cluster]
        ax.scatter(centroid[0], centroid[1], centroid[2],
                   color=cluster_colors[i % len(cluster_colors)], marker='D', s=40, label=f'Centroid {cluster}')

    # Add highlighted patterns to the plot with cluster colors and default size
    for i, row in selected_points.iterrows():
        cluster_idx = int(row['Cluster'])
        ax.scatter(row['PC1'], row['PC2'], row['PC3'],
                   color=cluster_colors[cluster_idx % len(cluster_colors)], s=60)

        # Add label above and centered
        ax.text(row['PC1'], row['PC2'], row['PC3'] + 0.09, row['Pattern'], fontsize=12,
                color=cluster_colors[cluster_idx % len(cluster_colors)], ha='center', va='bottom')

    # Add title and axis labels
    ax.set_title(f'3D Clustering Visualization with Centroids ({model} - {dataset} - {class_label})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # Show legend
    ax.legend(loc='upper left', ncol=2)

    # Save the figure as a PNG file
    fig.savefig(f'./results/{model}-{data_type}-{dataset}-{class_label}.png', dpi=600)

    # Display the plot
    plt.show()

    return selected_points, selection_types


# Function to plot 1D distances to centroids
def plot_1D(model, dataset, class_label, data_type='1D'):
    """
    Creates a 1D bar plot of distances to centroids and saves as PNG.

    Parameters:
    - model (str): Model name.
    - dataset (str): Dataset name.
    - class_label (int): Class label.
    - data_type (str): Type of analysis (default: '1D').
    """
    # Load the CSV file
    data = pd.read_csv(f'./results/{model}-3D-{dataset}-{class_label}.csv')

    # Exclude rows where the 'Pattern' is 'Centroid'
    data = data[data['Pattern'] != 'Centroid']

    # Get unique clusters
    clusters = sorted(data['Cluster'].unique())

    # Create lists to store grouped data
    all_labels = []
    all_distances = []
    cluster_indices = []  # To store cluster membership for each bar
    cluster_boundaries = []  # To store vertical line positions

    pos = 0  # Position tracker for bars

    # Populate lists with data from each cluster
    for cluster in clusters:
        cluster_data = data[data['Cluster'] == cluster]

        # Sort data by 'Distance_to_Centroid'
        sorted_data = cluster_data.sort_values(by='Distance_to_Centroid')
        sorted_labels = sorted_data['Pattern']
        sorted_distances = sorted_data['Distance_to_Centroid']

        # Store labels and distances
        all_labels.extend(sorted_labels)
        all_distances.extend(sorted_distances)
        cluster_indices.extend([cluster] * len(sorted_labels))

        # Mark the cluster boundary
        pos += len(sorted_labels)
        cluster_boundaries.append(pos)

    # Create the plot
    plt.figure(figsize=(10, 5))
    bar_width = 0.5
    bar_positions = np.arange(len(all_distances))
    bar_colors = [cluster_colors[cluster] for cluster in cluster_indices]

    plt.bar(bar_positions, all_distances, color=bar_colors, zorder=2, width=bar_width)

    # Add vertical lines to separate clusters
    for boundary in cluster_boundaries[:-1]:
        plt.axvline(x=boundary - 0.5, color='green', linestyle='--', linewidth=1.5, zorder=3)

    # Add rotated labels
    plt.xticks(bar_positions, all_labels, rotation=90, ha='left')

    # Plot configurations
    plt.xlabel('Pattern')
    plt.ylabel('Distance to Centroid')
    plt.grid(True, linestyle='--', linewidth=0.5, zorder=-1)

    # Add legend
    handles = [plt.Line2D([0], [0], color=cluster_colors[i], lw=4) for i in range(len(clusters))]
    labels = [f'Cluster {i}' for i in range(len(clusters))]
    plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1), fancybox=True, shadow=True, ncol=1)

    plt.tight_layout()

    # Save the figure
    image_name = f'./results/{model}-{data_type}-{dataset}-{class_label}.png'
    plt.savefig(image_name, dpi=600)
    plt.show()
