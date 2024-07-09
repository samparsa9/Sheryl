import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA



def plot_features(df):
    # Check if 'Cluster' column exists in the DataFrame
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column")
    
    # Standardize the features
    scaler = StandardScaler()
    features = df.columns.drop('Cluster')
    scaled_features = scaler.fit_transform(df[features])
    
    # List of classifiers for outlier detection
    classifiers = {
        'KNN': KNN(),
        'Isolation Forest': IForest(),
        'LOF': LOF()
    }
    
    # Select feature pairs to plot
    feature_pairs = [
        ('Daily $ Volume', '50 SMA % Difference'),
        ('200 SMA % Difference', '50 Day EMA % Difference'),
        ('200 Day EMA % Difference', 'Beta value'),
        ('Sharpe Ratio', 'Volatility')
    ]
    
    # Initialize the plot
    num_classifiers = len(classifiers)
    fig, axes = plt.subplots(len(feature_pairs), num_classifiers, figsize=(15, 5 * len(feature_pairs)))
    
    if len(feature_pairs) == 1:
        axes = [axes]
    
    for row, (feature_x, feature_y) in enumerate(feature_pairs):
        # Select the current pair of features
        X = df[[feature_x, feature_y]].values
        X_scaled = scaler.fit_transform(X)
        
        for col, (clf_name, clf) in enumerate(classifiers.items()):
            # Fit the model
            clf.fit(X_scaled)
            
            # Predict the results
            y_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
            y_scores = clf.decision_scores_  # raw outlier scores
            
            # Create a grid for contour plot
            xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 200),
                                 np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 200))
            zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            zz = zz.reshape(xx.shape)
            
            # Scatter plot with cluster and outlier information
            ax = axes[row][col]
            scatter = ax.scatter(X[:, 0], X[:, 1], c=df['Cluster'], cmap='viridis', label='Clusters', alpha=0.6, edgecolor='k')
            outliers = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', marker='x', label='Outliers', alpha=0.8)
            
            # Contour plot
            ax.contourf(xx, yy, zz, levels=np.linspace(zz.min(), zz.max(), 10), cmap='coolwarm', alpha=0.3)
            ax.contour(xx, yy, zz, levels=[0], linewidths=2, colors='red')
            
            # Density plot
            sns.kdeplot(x=X[:, 0], y=X[:, 1], ax=ax, fill=True, cmap="Blues", alpha=0.1)
            
            # Title and legend
            ax.set_title(f"{clf_name} Outlier Detection\nFeatures: {feature_x} vs {feature_y}")
            if col == 0:
                ax.set_ylabel(f"{feature_y}")
            if row == len(feature_pairs) - 1:
                ax.set_xlabel(f"{feature_x}")
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            legend2 = ax.legend(['Inliers', 'Outliers'], loc='upper right')
            ax.add_artist(legend2)
    
    # Show plot
    plt.suptitle("Outlier Detection with Different Algorithms", size=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_features_2(df):
    # Check if 'Cluster' column exists in the DataFrame
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column")
    
    # Standardize the features
    scaler = StandardScaler()
    features = df.columns.drop('Cluster')
    scaled_features = scaler.fit_transform(df[features])
    
    # List of classifiers for outlier detection
    classifiers = {
        'KNN': KNN(),
        'Isolation Forest': IForest(),
        'LOF': LOF()
    }
    
    # Initialize the plot
    num_classifiers = len(classifiers)
    fig, axes = plt.subplots(num_classifiers, figsize=(15, 5 * num_classifiers))
    
    if num_classifiers == 1:
        axes = [axes]
    
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # Fit the model
        clf.fit(scaled_features)
        
        # Predict the results
        y_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        
        # Scatter plot with cluster and outlier information
        ax = axes[i]
        scatter = ax.scatter(df[features[0]], df[features[1]], c=df['Cluster'], cmap='viridis', label='Clusters', alpha=0.6)
        outliers = ax.scatter(df[features[0]], df[features[1]], c=y_pred, cmap='coolwarm', marker='x', label='Outliers', alpha=0.8)
        
        # Density plot
        sns.kdeplot(x=df[features[0]], y=df[features[1]], ax=ax, fill=True, cmap="Blues", alpha=0.3)
        
        # Title and legend
        ax.set_title(f"{clf_name} Outlier Detection")
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        legend2 = ax.legend(['Inliers', 'Outliers'], loc='upper right')
        ax.add_artist(legend2)
    
    # Show plot
    plt.suptitle("Outlier Detection with Different Algorithms", size=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_features_1(df):
    # Check if 'Cluster' column exists in the DataFrame
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column")
    
    # Get the list of features excluding the 'Cluster' column
    features = df.columns.drop('Cluster')
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Features")
    plt.show()

    # Plot pair plot for selected features
    selected_features = ['Daily $ Volume', '50 SMA % Difference', '200 SMA % Difference', '50 Day EMA % Difference']  # Select top features
    sns.pairplot(df, vars=selected_features, hue="Cluster", palette="husl", markers=["o", "s", "D", "P", "X"])
    plt.suptitle("Pair Plot of Selected Features by Cluster", y=1.02)
    plt.show()

    # Display descriptive statistics
    print("Descriptive Statistics of Features:")
    print(df[features].describe().transpose())

def plot_features_3(df):
    # Check if 'Cluster' column exists in the DataFrame
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column")

    # Get the list of features excluding the 'Cluster' column
    features = df.columns.drop('Cluster')
    
    # Plot correlation matrix with cluster information
    plt.figure(figsize=(12, 10))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("Correlation Matrix with Clustering Information")
    plt.show()
    
    # Plot 3D scatter plot for three selected features
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Daily $ Volume'], df['50 SMA % Difference'], df['200 SMA % Difference'], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel('Daily $ Volume')
    ax.set_ylabel('50 SMA % Difference')
    ax.set_zlabel('200 SMA % Difference')
    plt.title("3D Scatter Plot of Selected Features by Cluster")
    plt.show()
    
    # Plot parallel coordinates plot
    plt.figure(figsize=(15, 10))
    pd.plotting.parallel_coordinates(df, 'Cluster', color=plt.cm.Set2.colors)
    plt.title("Parallel Coordinates Plot by Cluster")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.show()
    
    # Heatmap with clustering
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.groupby('Cluster').mean(), annot=True, cmap='coolwarm')
    plt.title("Heatmap of Mean Feature Values by Cluster")
    plt.show()
    
def plot_features_4(df):
    # Check if 'Cluster' column exists in the DataFrame
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column")
    
    # Standardize the features
    scaler = StandardScaler()
    features = df.columns.drop('Cluster')
    scaled_features = scaler.fit_transform(df[features])
    
    # List of classifiers for outlier detection
    classifiers = {
        'KNN': KNN(),
        'Isolation Forest': IForest()
        #'LOF': LOF()
    }
    
    # Initialize the plot
    num_classifiers = len(classifiers)
    fig, axes = plt.subplots(num_classifiers, 1, figsize=(15, 5 * num_classifiers))
    
    if num_classifiers == 1:
        axes = [axes]
    
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # Fit the model
        clf.fit(scaled_features)
        
        # Predict the results
        y_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        
        # Scatter plot with cluster and outlier information
        ax = axes[i]
        scatter = ax.scatter(df[features[0]], df[features[1]], c=df['Cluster'], cmap='viridis', label='Clusters', alpha=0.6)
        outliers = ax.scatter(df[features[0]], df[features[1]], c=y_pred, cmap='coolwarm', marker='x', label='Outliers', alpha=0.8)
        
        # Density plot
        sns.kdeplot(x=df[features[0]], y=df[features[1]], ax=ax, fill=True, cmap="Blues", alpha=0.3)
        
        # Title and legend
        ax.set_title(f"{clf_name} Outlier Detection")
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        legend2 = ax.legend(['Inliers', 'Outliers'], loc='upper right')
        ax.add_artist(legend2)
    
    # Show plot
    plt.suptitle("Outlier Detection with Different Algorithms", size=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_features_interactive(df):
    # Check if 'Cluster' column exists in the DataFrame
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column")

    # Ensure all features are numeric and drop rows with NaNs
    features = df.columns.drop('Cluster')
    
    # Debugging: Print the features before converting to numeric
    print("Features before conversion to numeric:", df[features].dtypes)
    
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    df = df.dropna()

    # Debugging: Check if DataFrame is empty after dropping NaNs
    if df.empty:
        raise ValueError("DataFrame is empty after converting features to numeric and dropping NaNs.")
    
    # Debugging: Print the first few rows of the DataFrame to verify
    print("DataFrame after conversion and dropping NaNs:")
    print(df.head())
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_features)
    df['PCA1'] = components[:, 0]
    df['PCA2'] = components[:, 1]
    
    # Debugging: Print PCA components to verify
    print("PCA components:")
    print(df[['PCA1', 'PCA2']].head())
    
    # Plot PCA scatter plot
    fig_pca = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', title="PCA of Features",
                         labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'})
    fig_pca.show()
    
    # Correlation heatmap
    corr = df[features].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    fig_corr.show()
    
    # 3D scatter plot for three selected features
    if 'Daily $ Volume' in features and '50 SMA % Difference' in features and '200 SMA % Difference' in features:
        fig_3d = px.scatter_3d(df, x='Daily $ Volume', y='50 SMA % Difference', z='200 SMA % Difference', 
                               color='Cluster', title="3D Scatter Plot of Selected Features")
        fig_3d.show()
    else:
        print("Skipping 3D scatter plot because required features are not present.")
    
    # Parallel coordinates plot
    fig_parallel = px.parallel_coordinates(df, color='Cluster',
                                           dimensions=features,
                                           title="Parallel Coordinates Plot")
    fig_parallel.show()
    
    # Heatmap with clustering
    cluster_means = df.groupby('Cluster').mean().reset_index()
    fig_cluster_heatmap = px.imshow(cluster_means.set_index('Cluster').T, text_auto=True, aspect="auto", title="Heatmap of Mean Feature Values by Cluster")
    fig_cluster_heatmap.show()

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

def plot_features_interactive_1(df):
    # Check if 'Cluster' column exists in the DataFrame
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column")

    # Ensure all features are numeric and drop rows with NaNs
    features = df.columns.drop('Cluster')
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    df = df.dropna()

    # Check if DataFrame is empty after dropping NaNs
    if df.empty:
        raise ValueError("DataFrame is empty after converting features to numeric and dropping NaNs.")
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    components = pca.fit_transform(df[features])
    df['PCA1'] = components[:, 0]
    df['PCA2'] = components[:, 1]

    # Interactive PCA Scatter Plot with Animation
    fig_pca = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', animation_frame='Cluster',
                         title="Interactive PCA Scatter Plot with Animation",
                         labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'})
    fig_pca.show()

    # Sunburst Chart for Cluster Composition
    fig_sunburst = px.sunburst(df, path=['Cluster'], values='PCA1', 
                               title="Sunburst Chart for Cluster Composition")
    fig_sunburst.show()

    # Radar Chart for Feature Comparison Across Clusters
    categories = features
    cluster_means = df.groupby('Cluster').mean().reset_index()
    
    fig_radar = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
    
    for i, row in cluster_means.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=row[features].values,
            theta=categories,
            fill='toself',
            name=f'Cluster {int(row["Cluster"])}'
        ))

    fig_radar.update_layout(title="Radar Chart for Feature Comparison Across Clusters")
    fig_radar.show()
    
    # Interactive Heatmap with Dendrogram
    fig_dendro = make_subplots(rows=1, cols=2, column_widths=[0.3, 0.7], 
                               specs=[[{"type": "scatter"}, {"type": "heatmap"}]])

    dendro = sch.dendrogram(sch.linkage(df[features], method='ward'), orientation='left', labels=df.index, ax=fig_dendro['layout']['xaxis1'])
    dendro_leaves = dendro['leaves']
    
    df_dendro = df.iloc[dendro_leaves]
    
    heat_data = df_dendro[features].values
    heatmap = go.Heatmap(z=heat_data, x=features, y=df_dendro.index, colorscale='Viridis')
    
    fig_dendro.add_trace(heatmap, row=1, col=2)
    fig_dendro.update_layout(title="Interactive Heatmap with Dendrogram")
    fig_dendro.show()
    
    # Interactive Violin Plot for Feature Distribution
    fig_violin = px.violin(df, y=features, x='Cluster', color='Cluster', box=True, points='all', 
                           title="Interactive Violin Plot for Feature Distribution")
    fig_violin.show()