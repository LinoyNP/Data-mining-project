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
    diff = x - centroid
    normalized_diff = diff / std
    squared = normalized_diff ** 2
    distance = np.sqrt(np.sum(squared))
    return distance

def InitializeTheFirst_K_Centroids(all_points, k):
    initialIndices = random.sample(all_points, k)
    return initialIndices

def RepresentClusterAsVector(points):
    N = len(points)
    SUM = np.sum(points, axis=0)
    SUMSQ = np.sum(points ** 2, axis=0)
    return N, SUM, SUMSQ

def UnionCluster(N1, SUM1, SUMSQ1, N2, SUM2, SUMSQ2):
    return N1 + N2, SUM1 + SUM2, SUMSQ1 + SUMSQ2

def computeCentroidAndStd(N, SUM, SUMSQ):
    centroid = SUM / N
    variance = (SUMSQ / N) - (centroid ** 2)
    std_dev = np.sqrt(np.maximum(variance, 1e-10))
    return centroid, std_dev

def bfr_cluster(dim, k, n, block_size, in_path, out_path):
    DS = {}
    CS = {}
    RS = []
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
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k).fit(unassigned_points)
            for i, label in enumerate(kmeans.labels_):
                cluster_points = np.array(unassigned_points)[kmeans.labels_ == label]
                if len(cluster_points) == 1:
                    RS.append(cluster_points[0])
                else:
                    N, SUM, SUMSQ = RepresentClusterAsVector(cluster_points)
                    CS[len(CS)] = (N, SUM, SUMSQ)

        final_clusters = {}
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