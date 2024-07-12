import pandas as pd
from sklearn.cluster import KMeans
# from .k_means_clustering import KMeans

# Function to apply k means clustering algorithm from sci-kit learn to the scaled data and add a cluster column
# to the dataframe passed in to specify the cluster for each symbol in the df
def apply_k_means(df, scaled_data, num_clusters):
    # kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    # clusters = kmeans.fit_predict(scaled_data)
    # df['Cluster'] = pd.Series(clusters, index=df.index)
    kmeans = KMeans(num_clusters)
    print(f"Scaled_data: {scaled_data}")
    clusters = kmeans.fit(scaled_data)
    df['Cluster'] = pd.Series(clusters,index = df.index )
