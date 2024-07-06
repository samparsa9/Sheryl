import pandas as pd
from sklearn.cluster import KMeans

def apply_k_means(df, scaled_data, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    df['Cluster'] = pd.Series(clusters, index=df.index)