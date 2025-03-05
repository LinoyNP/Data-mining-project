#Name: Linoy Nisim Pur  ID:324029685
#Name: Noa Shem Tov     ID:207000134
"-------------------------------------------------part 2 of 3-----------------------------------------------------------"
import csv
import os
import random
import part1
import numpy as np
from itertools import repeat, islice
from functools import partial

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

def computeCentroidPerClusterAndStd(N, SUM, SUMSQ):
    """
        Computes the centroid and standard deviation of a cluster based on its parameters.

        :param N: The number of points in the cluster.
        :param SUM: The sum of points in the cluster.
        :param SUMSQ: The sum of squared points in the cluster.
        :return: The centroid and standard deviation of the cluster.
        """
    centroid = SUM / N
    variance = (SUMSQ / N) - (centroid ** 2)
    std_dev = np.sqrt(np.maximum(variance, 1e-10))#  1e-10 to prevent Divide by zero
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
        centroid1, std1 = computeCentroidPerClusterAndStd(N1, SUM1, SUMSQ1)

        for j in range(i + 1, len(keys)):
            if keys[j] in merged_ids:
                continue
            N2, SUM2, SUMSQ2 = cluster_dict[keys[j]]
            centroid2, std2 = computeCentroidPerClusterAndStd(N2, SUM2, SUMSQ2)

            if Mahalanobis(centroid1, centroid2, std1) < threshold:
                N1, SUM1, SUMSQ1 = UnionCluster(N1, SUM1, SUMSQ1, N2, SUM2, SUMSQ2)
                merged_ids.add(keys[j])

        merged[new_id] = (N1, SUM1, SUMSQ1)
        merged_ids.add(keys[i])
        new_id += 1

    return merged

def MergePointToCluster(point,N,SUM,SUMSQ):
    arrayPoint = np.array(point)
    newN = 1 + N
    newSum = arrayPoint + SUM
    newSUMSQ = SUMSQ + arrayPoint**2
    return newN, newSum, newSUMSQ

# def bfr_cluster(dim, k, n, block_size, in_path, out_path):
#     DS = []
#     CS = []
#     RS = []
#     final_clusters = {}
#     NumOfIteration = (n // block_size) - 1
#     #iteration number 1
#     threshold = 2
#     AllPoints = []
#     part1.load_points(in_path, dim, block_size, AllPoints)  # אתחול הנקודות
#     AllPoints = [tuple(sublist) for sublist in AllPoints]
#     if k is None:
#         k = part1.FindKOptimal(dim, n, AllPoints)  # מציאת הK האופטימלי
#     initialClusters = part1.k_means(dim,k,block_size,AllPoints)
#     with open(out_path, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         for i in range(len(initialClusters)):
#             for point in initialClusters[i]:
#                 point = point+(i,)
#                 writer.writerow(point)
#     for list in initialClusters:
#         TheReducedPoints = RepresentClusterAsVector(np.array(list))
#         DS.append(TheReducedPoints)
#     flagForKnowingWhichClusterThePointBelongs = False
#     for iteration in range(NumOfIteration):
#         part1.load_points(in_path, dim, block_size, AllPoints)
#         for indexOfPOintInAllPoints,point in enumerate(AllPoints):
#             for ThePlaceOfClusterInDs, VectorInDs in enumerate(DS):
#                 N, SUM, SUMSQ = VectorInDs
#                 centroid,StdForVector = computeCentroidPerClusterAndStd(N, SUM, SUMSQ)
#                 if Mahalanobis(np.array(point), np.array(centroid), StdForVector) < threshold:
#                     DS[ThePlaceOfClusterInDs] = MergePointToCluster(point,N,SUM,SUMSQ)
#                     flagForKnowingWhichClusterThePointBelongs = True
#                     break
#             if not flagForKnowingWhichClusterThePointBelongs:
#                 RS.append(point)
#             AllPoints.remove(point)
#         #K-means on RS
#         if RS:
#             ClustersInRs , centroidInRs = part1.run_k_means(dim,k,len(RS),RS,100)
#             for indexOfCluster, currentClusterForCheck in enumerate(ClustersInRs):#all Clusters
#                 TheReducedPoints = RepresentClusterAsVector(np.array(currentClusterForCheck))
#                 N, SUM, SUMSQ = TheReducedPoints
#                 _, StdForVector = computeCentroidPerClusterAndStd(N, SUM, SUMSQ)
#                 for point in currentClusterForCheck:#specific cluster
#                     if Mahalanobis(np.array(point), np.array(centroidInRs[indexOfCluster]), StdForVector) > threshold:
#                         currentClusterForCheck.remove(point)#remove the point from the cluster because the threshold
#                 if len(currentClusterForCheck) > 1:
#                     with open("CS_FILE.csv", 'a', newline='') as csvfile:
#                         writer = csv.writer(csvfile)
#                         for point in currentClusterForCheck:
#                             RS.remove(point)
#                             point = point + (RepresentClusterAsVector(np.array(currentClusterForCheck)),)
#                             writer.writerow(point)
#                     CS.append(RepresentClusterAsVector(np.array(currentClusterForCheck)))
#
#

def bfr_cluster(dim, k, n, block_size, in_path, out_path):
    DS = {}  # {cluster_id: (N, SUM, SUMSQ)}
    CS = {}  # {cluster_id: (N, SUM, SUMSQ)}
    RS = []
    point_cluster_mapping = {}  # {point_index: cluster_id}
    cluster_id_counter = 0
    threshold = 2

    # Load first block
    AllPoints = []
    part1.load_points(in_path, dim, block_size, AllPoints)
    AllPoints = [tuple(sublist) for sublist in AllPoints]

    if k is None:
        k = part1.FindKOptimal(dim, n, AllPoints)

    initialClusters = part1.k_means(dim, k, block_size, AllPoints)

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i, cluster in enumerate(initialClusters):
            N, SUM, SUMSQ = RepresentClusterAsVector(np.array(cluster))
            DS[cluster_id_counter] = (N, SUM, SUMSQ)
            for point in cluster:
                writer.writerow(point + (cluster_id_counter,))
                point_cluster_mapping[tuple(point)] = cluster_id_counter
            cluster_id_counter += 1

    total_blocks = (n // block_size) - 1

    for _ in range(total_blocks):
        block_points = []
        part1.load_points(in_path, dim, block_size, block_points)
        for point in block_points:
            assigned = False
            for cluster_id, (N, SUM, SUMSQ) in DS.items():
                centroid, std = computeCentroidPerClusterAndStd(N, SUM, SUMSQ)
                if Mahalanobis(np.array(point), centroid, std) < threshold:
                    DS[cluster_id] = MergePointToCluster(point, N, SUM, SUMSQ)
                    point_cluster_mapping[tuple(point)] = cluster_id
                    assigned = True
                    break
            if not assigned:
                RS.append(point)

        if len(RS) >= 2 * k:
            RS_clusters, RS_centroids = part1.run_k_means(dim, k, len(RS), RS, 100)
            new_RS = []
            for i, cluster in enumerate(RS_clusters):
                if len(cluster) > 1:
                    N, SUM, SUMSQ = RepresentClusterAsVector(np.array(cluster))
                    CS[cluster_id_counter] = (N, SUM, SUMSQ)
                    for point in cluster:
                        point_cluster_mapping[tuple(point)] = cluster_id_counter
                    cluster_id_counter += 1
                else:
                    new_RS.extend(cluster)
            RS = new_RS
        CS = merge_clusters(CS, threshold)

    if len(RS) > 0:
        RS_clusters, RS_centroids = part1.run_k_means(dim, k, len(RS), RS, 100)
        for i, cluster in enumerate(RS_clusters):
            if len(cluster) > 1:
                N, SUM, SUMSQ = RepresentClusterAsVector(np.array(cluster))
                CS[cluster_id_counter] = (N, SUM, SUMSQ)
                for point in cluster:
                    point_cluster_mapping[tuple(point)] = cluster_id_counter
                cluster_id_counter += 1

    for cs_id, (N_cs, SUM_cs, SUMSQ_cs) in CS.items():
        cs_centroid, cs_std = computeCentroidPerClusterAndStd(N_cs, SUM_cs, SUMSQ_cs)
        merged = False
        for ds_id, (N_ds, SUM_ds, SUMSQ_ds) in DS.items():
            ds_centroid, ds_std = computeCentroidPerClusterAndStd(N_ds, SUM_ds, SUMSQ_ds)
            if Mahalanobis(cs_centroid, ds_centroid, ds_std) < threshold:
                DS[ds_id] = UnionCluster(N_ds, SUM_ds, SUMSQ_ds, N_cs, SUM_cs, SUMSQ_cs)
                merged = True
                break
        if not merged:
            DS[cluster_id_counter] = (N_cs, SUM_cs, SUMSQ_cs)
            cluster_id_counter += 1

    final_points = list(point_cluster_mapping.keys())
    final_clusters = []
    for cluster_id, (N, SUM, SUMSQ) in DS.items():
        centroid, _ = computeCentroidPerClusterAndStd(N, SUM, SUMSQ)
        final_clusters.append(centroid)

    final_assignments = part1.k_means(dim, k, len(final_points), final_points)

    with open(out_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for cluster_idx, cluster in enumerate(final_assignments):
            for point in cluster:
                writer.writerow(point + (cluster_idx,))

bfr_cluster(3, 2, 4661, 200, "out_path.csv", "blabla")

