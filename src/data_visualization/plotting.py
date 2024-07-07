import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lof import LOF


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