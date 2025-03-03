#Name: Linoy Nisim Pur  ID:324029685
#Name: Noa Shem Tov     ID:207000134
"-------------------------------------------------part 2 of 3-----------------------------------------------------------"
import csv
import random
import part1


def CreateMaxDate(DIM):
    """
       Generates large amounts of data by calling the `generate` function multiple times.
       The number of times `generate` is called is determined randomly within a range of 10 to 21 times.
       This function generates random datasets, where each dataset has a random number of points
       and a calculated value for `k`, and appends the generated data to the "out_path.csv" file.
       Parameters:
           DIM (int): The number of dimensions for the generated points. This value is passed to the
                      `generate_data` function to specify how many dimensions each point will have.
       Returns: None
       Requirements:
           The function will call `part1.generate_data` multiple times, where each call generates a dataset with:
               - A random number of points (between 30 and 1000).
               - A calculated value for `k` (the number of clusters, `k = currentN // 10`).
               - The output is saved to "out_path.csv".
       Notes:
           - The number of times `generate` is called is randomized between 10 and 21 times.
           - The number of points (`currentN`) for each dataset is also randomized between 30 and 1000.
           - Each dataset will have a corresponding number of clusters (`currentK`), calculated as `currentN // 10`.
       """
    theNumberOfTimeTheGenerateFunctionIsCalled = random.randint(10,21)
    for i in range(theNumberOfTimeTheGenerateFunctionIsCalled):
        currentN = random.randint(30,1000)
        currentK = currentN // 10
        part1.generate_data(3, currentK, currentN, "out_path.csv", points_gen=part1.creatingPoints, extras = {})

import numpy as np

def Mahalanobis(x, centroid, std):
    """
    Computes the Mahalanobis distance of a point from the centroid of a dataset using the standard deviation.

    :param x: A data point (vector).
    :param centroid: The centroid of the dataset (vector).
    :param std: The standard deviation of the dataset (vector).
    :return: The Mahalanobis distance between the point and the centroid.
    """
    diff = x - centroid
    normalized_diff = diff / std
    squared = normalized_diff ** 2
    distance = np.sqrt(np.sum(squared))
    return distance

def InitializeTheFirst_K_Centroids(all_points, k):
    """
        Initializes the first k centroids by randomly selecting points from the input data.

        :param all_points: A list of all points.
        :param k: The number of centroids to initialize.
        :return: A list of indices of the randomly selected centroids.
    """
    initialIndices = random.sample(list(all_points), k)
    return initialIndices

def RepresentClusterAsVector(points):
    """
        Represents a cluster of points as a vector, including the sum of points and the sum of squared points.

        :param points: A list of points.
        :return: The number of points, the sum of the points, and the sum of squared points.
        """
    N = len(points)
    SUM = np.sum(points, axis=0)
    SUMSQ = np.sum(points ** 2, axis=0)
    return N, SUM, SUMSQ

def UnionCluster(N1, SUM1, SUMSQ1, N2, SUM2, SUMSQ2):
    """
        Merges two clusters and returns the parameters of the merged cluster.

        :param N1, N2: The number of points in each cluster.
        :param SUM1, SUM2: The sum of points in each cluster.
        :param SUMSQ1, SUMSQ2: The sum of squared points in each cluster.
        :return: The parameters of the merged cluster: the total number of points, the sum of points, and the sum of squared points.
        """
    return N1 + N2, SUM1 + SUM2, SUMSQ1 + SUMSQ2

def computeCentroidAndStd(N, SUM, SUMSQ):
    """
        Computes the centroid and standard deviation of a cluster based on its parameters.

        :param N: The number of points in the cluster.
        :param SUM: The sum of points in the cluster.
        :param SUMSQ: The sum of squared points in the cluster.
        :return: The centroid and standard deviation of the cluster.
        """
    centroid = SUM / N
    variance = (SUMSQ / N) - (centroid ** 2)
    std_dev = np.sqrt(np.maximum(variance, 1e-10))
    return centroid, std_dev


def merge_clusters(cluster_dict, threshold=2):
    """
        Merges clusters if the Mahalanobis distance between their centroids is less than the specified threshold.

        :param cluster_dict: A dictionary of clusters with their parameters (N, SUM, SUMSQ).
        :param threshold: The distance threshold (default is 2).
        :return: A dictionary of merged clusters.
        """
    merged = {}
    keys = list(cluster_dict.keys())
    merged_ids = set()
    new_id = 0

    for i in range(len(keys)):
        if keys[i] in merged_ids:
            continue
        N1, SUM1, SUMSQ1 = cluster_dict[keys[i]]
        centroid1, std1 = computeCentroidAndStd(N1, SUM1, SUMSQ1)

        for j in range(i + 1, len(keys)):
            if keys[j] in merged_ids:
                continue
            N2, SUM2, SUMSQ2 = cluster_dict[keys[j]]
            centroid2, std2 = computeCentroidAndStd(N2, SUM2, SUMSQ2)

            if Mahalanobis(centroid1, centroid2, std1) < threshold:
                N1, SUM1, SUMSQ1 = UnionCluster(N1, SUM1, SUMSQ1, N2, SUM2, SUMSQ2)
                merged_ids.add(keys[j])

        merged[new_id] = (N1, SUM1, SUMSQ1)
        merged_ids.add(keys[i])
        new_id += 1

    return merged

def bfr_cluster(dim, k, n, block_size, in_path, out_path):
    """
        Performs clustering using the BFR algorithm, with centroid and standard deviation computation and merging of clusters based on Mahalanobis distance.

        :param dim: The number of dimensions of the points.
        :param k: The desired number of clusters (if None, it will be computed automatically).
        :param n: The number of points to cluster.
        :param block_size: The number of points to load into memory at once.
        :param in_path: The input file path containing the points.
        :param out_path: The output file path where the results will be saved.
        :return: None. The results are saved to the output file.
        """
    DS = {}
    CS = {}
    RS = []
    final_clusters = {}
    for i in range(n // block_size):
        AllPoints = []
        part1.load_points(in_path, dim, block_size, AllPoints)
        AllPoints = np.array(AllPoints)
        if k is None:
            k = part1.FindKOptimal(dim,n,AllPoints)
        initialCentroids = InitializeTheFirst_K_Centroids(AllPoints, k)
        unassigned_points = []
        for point in AllPoints:
            assigned = False
            for cluster_id, (N, SUM, SUMSQ) in DS.items():
                centroid, std_dev = computeCentroidAndStd(N, SUM, SUMSQ)
                if Mahalanobis(point, centroid, std_dev) < 2:
                    DS[cluster_id] = UnionCluster(N, SUM, SUMSQ, 1, point, point ** 2)
                    assigned = True
                    break
            if not assigned:
                unassigned_points.append(point)

        if RS:
            unassigned_points.extend(RS)
        RS = []

        if len(unassigned_points) >= k:
            clusters, centroids = part1.run_k_means(dim, k, n, unassigned_points, 100)
            for i, cluster in enumerate(clusters):
                cluster_points = np.array(cluster)
                if len(cluster_points) == 1:
                    RS.append(cluster_points[0])
                else:
                    N, SUM, SUMSQ = RepresentClusterAsVector(cluster_points)
                    CS[len(CS)] = (N, SUM, SUMSQ)
            CS = merge_clusters(CS)

            for cs_id, (N, SUM, SUMSQ) in list(CS.items()):
                for ds_id, (N_ds, SUM_ds, SUMSQ_ds) in list(DS.items()):
                    centroid_cs, std_cs = computeCentroidAndStd(N, SUM, SUMSQ)
                    centroid_ds, std_ds = computeCentroidAndStd(N_ds, SUM_ds, SUMSQ_ds)

                    if Mahalanobis(centroid_cs, centroid_ds, std_ds) < 2:
                        DS[ds_id] = UnionCluster(N_ds, SUM_ds, SUMSQ_ds, N, SUM, SUMSQ)
                        del CS[cs_id]
                        break
        cluster_id = 0

        for cluster in DS.values():
            final_clusters[cluster_id] = cluster
            cluster_id += 1

        for cluster in CS.values():
            final_clusters[cluster_id] = cluster
            cluster_id += 1

        for point in RS:
            final_clusters[cluster_id] = (1, point, point ** 2)
            cluster_id += 1

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [f'x{i + 1}' for i in range(dim)] + ['cluster_id']
        writer.writerow(header)

        for cid, (N, SUM, SUMSQ) in final_clusters.items():
            centroid, _ = computeCentroidAndStd(N, SUM, SUMSQ)
            for _ in range(N):
                writer.writerow(list(centroid) + [cid])