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
    """Compute the Euclidean distance between two points in any dimension."""
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
