import csv
import random
import numpy as np
import math


def generate_data(dim, k, n, out_path, points_gen=None, extras = {}):
    """
    Generates a CSV file containing data points divided into k clusters within a space of dim dimensions.

    :param dim: The number of dimensions in the space where the points will be generated.
    :param k: The number of clusters to which the data points will be assigned.
    :param n: The total number of data points to generate.
    :param out_path: The file path where the generated CSV will be saved.
    :param points_gen: (Optional) A function or method to generate the data points
                       (or part of them) according to a custom distribution.
    :param extras: (Optional) A dictionary containing additional parameters for further customization.
    :return: None

    Note:
    Each iteration simulates the selection of a group of points.
    The points themselves are not explicitly stored as fixed clusters,
    but in this way, each iteration produces several points that may form a potential cluster.
    """

    arrayOfGroupSizes = np.random.multinomial(n, [1/k] * k)  # Divides n among k groups randomly so that the sum of the sizes is exactly n.

    maxNum = 10000  # The maximum number that can be in a group
    minNum = 5000  # The minimum number that can be in a group

    with open(out_path, "w", newline="") as file:
        writer = csv.writer(file)
        for i in range(k):
            maxMumberInThisGroup = random.randint(minNum, maxNum)  # The maximum number that can be in this group
            groupOfPoints = points_gen(dim, arrayOfGroupSizes[i], maxMumberInThisGroup)
            writer.writerows(groupOfPoints)
            # for point in groupOfPoints:
            #     file.write(f" {point[:]}\n")

def creatingPoints (dim, numberOfPoints, maxNum):
    """
    Creating points for one iteration
    :param dim: The number of dimensions in the space where the points will be generated.
    :param numberOfPoints: number of points that will be in this group
    :param maxNum: The maximum number that can be in this group
    :return: list of points
    """
    points = []
    #Creating all the points that will be in the group
    for i in range(numberOfPoints):
        onePoint = ()
        #Creation of one point
        for j in range(dim):
            onePoint = onePoint + (random.randint(maxNum//2, maxNum), )
        points.append(onePoint)
    return points

def load_points(in_path, dim, n=-1, points=[]):
    with open(in_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        counter = 0
        for row in reader:
            if n != -1 and counter >= n:
                break
            points.append([float(x) for x in row[:dim]])
            counter += 1


# generate_data(3, 4, 50, "out_path.csv", points_gen=creatingPoints, extras = {})
def euclideanDistance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    :param p1: A list or tuple representing the first point.
    :param p2: A list or tuple representing the second point.
    :return: The Euclidean distance between p1 and p2.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def FindClosestClusters(clusters, DistanceParameterFunc):
    """Find the pair of clusters with the smallest distance between them."""
    CurrentMinDist = float('inf')
    mostClosePair = None
    for i, ClusterOne in enumerate(clusters):
        for j, ClusterTwo in enumerate(clusters):
            if i < j:
                d = DistanceParameterFunc(np.mean(list(ClusterOne), axis=0), np.mean(list(ClusterTwo), axis=0))
                if d < CurrentMinDist:
                    CurrentMinDist = d
                    mostClosePair = (i, j)

    return mostClosePair

def h_clustering(dim, k, points, dist=None, clusts=[]):
    """Perform bottom-up hierarchical clustering.

    Args:
        dim (int): Number of dimensions of the points.
        k (int or None): Desired number of clusters. If None, automatically determine when to stop.
        points (list of tuples): List of points to cluster.
        dist (function, optional): Distance function. Defaults to Euclidean distance.
        clusts (list, optional): Output list to store clusters. Defaults to an empty list.

    Returns:
        list: List of clusters, where each cluster is a list of points.
    """
    if dist is None:
        dist = euclideanDistance
    clusters = [[p] for p in points]
    while k is None or len(clusters) > k:
        mostClosePair = FindClosestClusters(clusters, dist)
        if mostClosePair is None:
            break
        i, j = mostClosePair
        newCluster = clusters[i] + clusters[j]
        clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)]
        clusters.append(newCluster)
        if k is None and len(clusters) <= 1:
            break
    clusts.extend([list(cluster) for cluster in clusters])
    return clusts

def calculateSSE(centroids, clusters):
    """
    Calculate the Sum of Squared Errors (SSE) for the current clustering.

    :param centroids: The list of centroids for each cluster.
    :param clusters: A list of clusters, each containing the points assigned to that cluster.
    :return: The SSE value.
    """
    sse = 0
    for i, cluster in enumerate(clusters):
        centroid = centroids[i]
        for point in cluster:
            sse += euclideanDistance(point, centroid) ** 2
    return sse

def k_means(dim, k, n, points, clusts=[]):
    """
    Perform K-Means clustering.

    :param dim: The number of dimensions for each point.
    :param k: The number of clusters to form (if None, the algorithm will find the optimal k).
    :param n: The number of points.
    :param points: A list of points (each a list or tuple of length 'dim').
    :param clusts: A list to store the resulting clusters.
    :return: A list of clusters containing points and the optimal k if k is None.
    """
    max_iterations = 100
    if k is None:
        MinSSE = float('inf')
        bestK = None

        # Try multiple values for k
        for currentK in range(2, n//10):  # We start from k=2 because k=1 is not a valid cluster count
            CurrentClusters, CurrentCentroid = run_k_means(dim, currentK, n, points, max_iterations)
            currentSSE = calculateSSE (CurrentCentroid,CurrentClusters)
            if currentSSE < MinSSE:
                MinSSE = currentSSE
                bestK = currentK
        k = bestK

    # Run K-Means with the given k
    clustersBeforeAssignment, centroid = run_k_means(dim, k, n, points, max_iterations)
    for clust in clustersBeforeAssignment:
        clusts.append(clust)
    return clusts

def run_k_means(dim, k, n, points, max_iterations):
    """
    Perform K-Means clustering for a specific value of k.

    :param dim: The number of dimensions for each point.
    :param k: The number of clusters to form.
    :param n: The number of points.
    :param points: A list of points (each a list or tuple of length 'dim').
    :param max_iterations: The maximum number of iterations to perform.
    :return: A tuple containing the clusters, centroids
    """

    # Randomly initialize centroids from the points
    centroids = random.sample(points, k)
    clusters = None
    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]

        # Step 1: Assign points to the nearest centroid
        for point in points:
            distances = [euclideanDistance(point, centroid) for centroid in centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(point)
        # Step 2: Recalculate centroids
        new_centroids = []
        for cluster in clusters:
            new_centroid = np.mean(cluster, axis=0).tolist()
            new_centroids.append(new_centroid)
        # Step 3: Check for convergence (if centroids do not change)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters,centroids
