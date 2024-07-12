import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import time as tm
from sklearn.datasets import make_blobs
import pandas as pd

counter = 0
class KMeans():
    def __init__(self, k):
        self.num_clusters = k
        self.cluster_dict = {}
        self.centroid_mean_diff_list = []
        self.data = None
        self.num_features = None
        self.df = False

    def fit(self, data_coords):
        data_coords = pd.DataFrame(data_coords)
        if isinstance(data_coords, pd.DataFrame):
            data_coords_df = data_coords.copy()
            data_coords = data_coords.values
            data_coords_df["cluster"] = None
            self.df = True
        self.data = data_coords
        self.num_features = self.data.shape[1]
        for cluster in range(self.num_clusters):
            # Cluster dictionary will have the key being the cluster # and the value being a list of a tuple of the coordiantes
            # of the cluster point, and a list of the data points in that cluster, initialized to an empty list for now
            #self.cluster_dict[cluster][0] will give the (x,y) coordinate of the cluster centroid
            # self.cluster_dict[cluster][1] will return a list of [x y] cordinates
            cluster_coordinates = []
            for i in range(0, self.num_features):
                feature_column_mean = np.mean(self.data[:,i]) 
                cluster_coordinates.append(feature_column_mean + 1) # +1 somehow fixes everything
                self.cluster_dict[cluster] = [np.array(cluster_coordinates), []]
            # x_mean = np.mean(self.data[:, 0])
            # y_mean = np.mean(self.data[:, 1])
            # self.cluster_dict[cluster] = [np.array([x_mean+1, y_mean+1]), []]
        
        plt.ion()
        fig = plt.figure(figsize=(12,8))
        if self.num_features == 2:
            ax1 = fig.add_subplot(111)
        elif self.num_features == 3:
            ax1 = fig.add_subplot(211, projection='3d')
        # Assiging datapoints to clusters
        while np.sum(self.centroid_mean_diff_list) > 0.01 or self.centroid_mean_diff_list == []: #or any(self.cluster_dict[i][1] == [] for i in list(self.cluster_dict.keys())):
            #self.live_plotting(ax1)
            # refreshing points in the clusters
            for i in list(self.cluster_dict.keys()):
                self.cluster_dict[i][1] = []
            # refreshing the list
            self.centroid_mean_diff_list = []
            # For every point in our dataset
            for j, data_coord in enumerate(data_coords):
                # the min cluster is set to nothing
                closest_cluster_distance = float('inf')
                closest_cluster = None
                # For each cluster, see how far away it is, if it is closer than our current min distance, update it to be the new
                # closest cluster
                for i in list(self.cluster_dict.keys()):
                    cluster_coord = self.cluster_dict[i][0] # (x,y,z,...)
                    squared_dist = 0
                    for feature in range(0, self.num_features):
                        squared_dist += (cluster_coord[feature] - data_coord[feature]) ** 2
                    distance = np.sqrt(squared_dist)
                    if distance < closest_cluster_distance:
                        closest_cluster_distance = distance
                        closest_cluster = i
                self.cluster_dict[closest_cluster][1].append(data_coord)
                if self.df:
                    data_coords_df.at[j, "cluster"] = closest_cluster

            # Recalculate the mean of the cluster centroids
            print(f"Redefining centroids based on mean of datapoints currently in cluster...")
            for i in list(self.cluster_dict.keys()):
                old_centroid_mean = np.array(self.cluster_dict[i][0]) # This is a (x,y,z,...) point
                print(f"Old Centroid Mean Point: {self.cluster_dict[i][0]}")
                if len(self.cluster_dict[i][1]) > 0:
                    self.cluster_dict[i][0] = np.mean(self.cluster_dict[i][1], axis=0) # The new coordinate of our centroid based on the mean of the data points in it
                new_centroid_mean = np.array(self.cluster_dict[i][0])# This is an (x,y,z...) point 
                print(f"New Centroid Mean Point: {new_centroid_mean}")
                try:
                    squared_dist = 0
                    for feature in range(0, self.num_features):
                        squared_dist += ((new_centroid_mean[feature] - old_centroid_mean[feature]) ** 2)# + ((new_centroid_mean[feature+1] - old_centroid_mean[feature+1]) ** 2)
                    distance = np.sqrt(squared_dist)
                    mean_change = distance
                except:
                     mean_change = 0
                print(f"mean_change: {mean_change}")
                self.centroid_mean_diff_list.append(mean_change)
                #tm.sleep(1)
            print("Finished updating means of centroids")
            print(f"List of new differences of means: {self.centroid_mean_diff_list}")
            print(f"sum: {np.sum(self.centroid_mean_diff_list)}")
            

        print("DONE")
        # plt.ioff()
        # plt.show()
        if self.df:
            return data_coords_df['cluster'].values
        else:
            return self.cluster_dict
    
    
    def live_plotting(self, ax1):
        ax1.cla()
        colors = plt.cm.get_cmap('viridis', self.num_clusters)

        if self.num_features == 2:
            for i in list(self.cluster_dict.keys()):
                cluster_points = np.array(self.cluster_dict[i][1])
                if len(cluster_points) > 0:
                    ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(i), label=f'Cluster {i}')

            cluster_x_coords = [self.cluster_dict[i][0][0] for i in list(self.cluster_dict.keys())]
            cluster_y_coords = [self.cluster_dict[i][0][1] for i in list(self.cluster_dict.keys())]
            ax1.scatter(cluster_x_coords, cluster_y_coords, color='red', marker='x', label='Centroids')

            plt.axis([-15, 15, -15, 15])
            plt.legend()
        elif self.num_features == 3:
            for i in list(self.cluster_dict.keys()):
                cluster_points = np.array(self.cluster_dict[i][1])
                if len(cluster_points) > 0:
                    ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=colors(i), label=f'Cluster {i}')

            cluster_x_coords = [self.cluster_dict[i][0][0] for i in list(self.cluster_dict.keys())]
            cluster_y_coords = [self.cluster_dict[i][0][1] for i in list(self.cluster_dict.keys())]
            cluster_z_coords = [self.cluster_dict[i][0][2] for i in list(self.cluster_dict.keys())]
            ax1.scatter(cluster_x_coords, cluster_y_coords, cluster_z_coords, color='red', marker='x', label='Centroids')

            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            plt.legend()

        plt.draw()
        plt.pause(0.25)
            
        
        


def main():
#     data = np.array([
#     [5, 3, 4],     # Eeach row is a datapoint, each column is a feature, since this is unsurpervised learning both the x and y
#     [1, 5, 6],     # coordinates will be features
#     [2, 8, 8], 
#     [3, 6, 2],
#     [6, 2, 9], 
#     [8, 1, 5], 
#     [7, 3, 1], 
#     [9, 4, 7],
#     [7, 7, 4], 
#     [6, 6, 8]
# ])
    X, y = make_blobs(n_samples=300, centers=4, n_features=3, random_state=counter) # random_state  doesnt work
    data = X

    df = pd.DataFrame(data)
    # Plot the dataset
    model = KMeans(4)
    print(model.fit(df))

    # plt.figure(figsize=(12,8))
    # plt.scatter(data[:, 0], data[:, 1], marker='o')
    # plt.show()


if __name__ == "__main__":
    #for i in range(100):
        main()
        counter += 1

