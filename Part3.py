#Name: Linoy Nisim Pur  ID:324029685
#Name: Noa Shem Tov     ID:207000134
"-------------------------------------------------part 3 of 3-----------------------------------------------------------"
import random

import numpy as np

import part1
import Part2
import pandas as pd
from collections import Counter

# $$$$$$$$$$$$$$$$$$  part one - small data
def CreateSmallDate(dataPath):
    """
    The CreateSmallDate function generates a small dataset and saves it to a CSV file. It randomly selects values for N
    (total data points) and K (subset size), then calls the generate_data function from an external module to create
    and store the data.
    currentN: A random integer between 1000 and 3000, representing the total number of data points.
    currentK: A random integer between 5 and currentN // 10, representing the subset size

    Call generate_data:
    dim = 3 – Specifies the number of dimensions for the generated data.
    currentK – The size of the data subset.
    currentN – The total number of data points.
    "smallDataPath.csv" – The output file where the data is stored.
    points_gen=part1.creatingPoints – A function used to generate data points.
    extras={} – An optional dictionary for additional parameters.

    Requirements
    The random module must be imported before calling this function.
    The part1 module must include the generate_data and creatingPoints functions.
    The dataset is stored in a CSV file named smallDataPath.csv.
    :dataPath: The path of the file that contains data
    :return: None
    """
    currentN = 1000
    currentK = random.randint(5,currentN // 10)
    part1.generate_data(2, currentK, currentN, dataPath, points_gen=part1.creatingPoints, extras={})


def generateValid_K1AndDim():
    """
     Generates a valid pair of values for `dim1` (number of dimensions) and `k1` (number of clusters) such that:
     - `dim1` is a random integer between 4 and 20 (inclusive), representing the number of dimensions.
     - `k1` is a random integer between 5 and 20 (inclusive), representing the number of clusters.
     - The sum of `dim1` and `k1` must be greater than 10.

     The function will continue generating random pairs until a valid pair is found,
     then return the values of `dim1` and `k1` as a tuple.

    :rtype: tuple
    :return:A tuple containing the values of `dim1` and `k1` that satisfy the condition.
    """
    while True:
        dim1 = random.randint(4, 20)  # dim1 must be greater than 3
        k1 = random.randint(5, 20)  # k1 must be greater than 4

        if dim1 + k1 > 10:
            return dim1, k1
def CreateDateWith_K_clustersAnd_Dim(dataPath):
    """

    Creates a data file small enough to work in main memory with generated points based on a specified number of dimensions and clusters.

    The function first determines valid values for the number of dimensions (dim1) and the number of clusters (k1),
    ensuring they satisfy the following conditions:
    - dim1 > 3
    - k1 > 4
    - dim1 + k1 > 10

    Then, it generates a random number of points within a range proportional to the number of clusters.
    Finally, it calls `part1.generate_data` to create and save the data to the specified path.

    Args:
        dataPath: (str) The file path where the generated data will be saved.
    Returns:
        None
    """
    dim1, k1 = generateValid_K1AndDim()
    n = random.randint(1000, 1500)
    part1.generate_data(dim1, k1, n, dataPath, points_gen=part1.creatingPoints, extras={})
    return dim1

def evaluateAssessingQualityOfClustering(clusters):
    centroids = []
    misclassifiedCount = 0
    numberofPoints = sum(len(cluster) for cluster in clusters)  # Total number of points
    # Calculate centroids for the resulting clusters
    for cluster in clusters:
        centroids.append(part1.np.mean(cluster, axis=0).tolist())

    for clusterIndex, cluster in enumerate(clusters):  # Iterate over each cluster
        for point in cluster:
            assigned_centroid = centroids[clusterIndex]  # The centroid of the assigned cluster

            # Compute distances from the point to all centroids
            distances = [part1.euclideanDistance(point, centroid) for centroid in centroids]

            # Find the index of the closest centroid
            closestCentroidIndex = part1.np.argmin(distances)

            # If the closest centroid does not belong to the assigned cluster → misclassified point
            if closestCentroidIndex != clusterIndex:
                misclassifiedCount += 1

    return misclassifiedCount / numberofPoints  # Return the misclassification rate


def compare_clustering_results(original_file, generated_file):
    """
    Compare the original clustering file with the generated clustering file and return the success rate.

    Since cluster numbers have no direct meaning, we must map clusters from the original file to those in the generated file.
    The mapping is based on majority voting: for each cluster in the original file, we find the most common corresponding cluster in the generated file.

    Steps:
    1. Load data from both files and ensure each point is assigned to a cluster.
    2. Create a mapping of points to clusters for both files.
    3. Find the best matching clusters by identifying the most frequently associated cluster in the generated file for each cluster in the original file.
    4. Compute the success rate as the percentage of correctly mapped points.

    :param original_file: Path to the original clustering file.
    :param generated_file: Path to the generated clustering file.
    :return: Success rate as a percentage.
    """

    df1 = pd.read_csv(original_file, header=None)
    df2 = pd.read_csv(generated_file, header=None)

    points1, clusters1 = df1.iloc[:, :-1].values, df1.iloc[:, -1].values
    points2, clusters2 = df2.iloc[:, :-1].values, df2.iloc[:, -1].values

    point_map1 = {tuple(p): c for p, c in zip(points1, clusters1)}
    point_map2 = {tuple(p): c for p, c in zip(points2, clusters2)}

    cluster_mapping = {}
    for cluster in set(clusters1):
        points_in_cluster1 = {p for p, c in point_map1.items() if c == cluster}
        matching_clusters = [point_map2[p] for p in points_in_cluster1 if p in point_map2]

        if matching_clusters:
            most_common_cluster = Counter(matching_clusters).most_common(1)[0][0]
            cluster_mapping[cluster] = most_common_cluster

    correct = sum(1 for p, c in point_map1.items() if p in point_map2 and point_map2[p] == cluster_mapping.get(c, -1))
    accuracy = correct / len(point_map1) * 100

    return accuracy

def runAlgorithmsForEachFileAndShowResults(original_file, dataFrame, dim):
    points_loaded = []
    part1.load_points(original_file, dim, -1, points_loaded)
    for i in range(2,9):
        try:
            clusts = []
            clusters = part1.k_means(dim, i, len(points_loaded), points_loaded, clusts)
            value = evaluateAssessingQualityOfClustering(clusters)
            dataFrame.loc[1, f"k={i}"] = value
            part1.ConvertClustersToFileInOrderByClustersIndex(clusts, f"results_kMeans_{original_file}_k={i}.csv")
            print(f"Updated: dataFrameForPartOne[k-means, 'k={i}'] = {value}")

            clusts = []
            part1.h_clustering(dim, i, points_loaded, None, clusts)
            value = evaluateAssessingQualityOfClustering(clusts)
            dataFrame.at[0, f"k={i}"] = value
            part1.ConvertClustersToFileInOrderByClustersIndex(clusts, f"results_hierarchical_{original_file}_k={i}.csv")
            print(f"Updated: dataFrameForPartOne[hierarchical, 'k={i}'] = {value}")
        except AssertionError as e:
            print(f"\n❌ error: {e}")
    print(f"\nQuality Index for File {original_file}")
    print(dataFrame.to_string(index=False))

def Compare2FilesAndDisplayResults(original_file, dataFrame):
    for i in range(2,9):
        try:
            accuracy = compare_clustering_results(original_file, f"results_kMeans_{original_file}_k={i}.csv")
            dataFrame.loc[1, f"k={i}"] = accuracy
            print(f"Updated: dataFrameForComparison[k-means, 'k={i}'] = {accuracy}%")

            accuracy = compare_clustering_results(original_file, f"results_hierarchical_{original_file}_k={i}.csv")
            dataFrame.at[0, f"k={i}"] = accuracy
            print(f"Updated: dataFrameForComparison[hierarchical, 'k={i}'] = {accuracy}%")
        except AssertionError as e:
            print(f"\n❌ error: {e}")
    print(f"\nComparing the cluster with respect to the original distribution in file {original_file}")
    print(dataFrame.to_string(index=False))


namesOfFiles = [f"file_{i}.csv" for i in range(2)]
dimsForEachFile = []

# Creating small data file
CreateSmallDate(namesOfFiles[0])
dimsForEachFile.append(2)

# Creating data files
for i in range(1, len(namesOfFiles)):
    dimsForEachFile.append(CreateDateWith_K_clustersAnd_Dim(namesOfFiles[i]))

# Checking the quality of the clusters
titleOfcolumns = ['File name', 'Algorithm', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7', 'k=8']
TableContents = []

# Creating DataFrame for each file
for file in namesOfFiles:
    TableContents.append([file, 'Hierarchical'] + [''] * 7)
    TableContents.append([file, 'k-means'] + [''] * 7)
dataFrameForPartOne = pd.DataFrame(TableContents, columns=titleOfcolumns)
# Running k-means and hierarchical algorithm for different K, and for small file and medium file
for i in range(len(namesOfFiles)):
    runAlgorithmsForEachFileAndShowResults(namesOfFiles[i], dataFrameForPartOne, dimsForEachFile[i])

dataFrameForPartOne = pd.DataFrame(TableContents, columns=titleOfcolumns)

# Adding the file name to the results in the table
def runAlgorithmsForEachFileAndShowResults(fileName, dataFrame, dims):
    for algo in ['Hierarchical', 'k-means']:
        for k in range(2, 9):
            # Append results to the DataFrame with the file name and algorithm name
            dataFrame.loc[dataFrame['File name'] == fileName, f'k={k}'] = f'Results for {fileName} with {algo}'

# Comparison between the source file and the result files
dataFrameForComparison = pd.DataFrame(TableContents, columns=titleOfcolumns)
for i in range(len(namesOfFiles)):
    Compare2FilesAndDisplayResults(namesOfFiles[i], dataFrameForComparison)

# Adding the file name to the comparison results in the table
def Compare2FilesAndDisplayResults(fileName, dataFrame):
    for algo in ['Hierarchical', 'k-means']:
        # Append comparison results to the DataFrame with the file name and algorithm name
        dataFrame.loc[dataFrame['File name'] == fileName, 'Comparison Results'] = f'Comparison results for {fileName} with {algo}'




# $$$$$$$$$$$$$$$$$$  part two - big data
def calculate_n(file_size_gb=10):
    """
        Calculates the required number of data points (n) to generate a file of at least `file_size_gb` GB.

        The calculation is based on the assumption that:
        - Each data point consists of `dim1` numerical values (floats).
        - Each float is stored as `float64` (8 bytes).
        - An additional label (cluster ID) is included, adding another 8 bytes per data point.
        - The total size of a single data point is `(dim1 + 1) * 8` bytes.

        Given a desired file size in bytes, `n` is computed as:
            n = total_bytes / bytes_per_point

        :param file_size_gb: Desired file size in GB (default is 10GB).
        :return: A tuple (dim1, k1, n) where:
                 - `dim1`: Number of dimensions.
                 - `k1`: Number of clusters.
                 - `n`: The calculated number of data points needed.
        """
    dim = random.randint(5, 20)
    k = random.randint(5, 20)
    bytes_per_point = (dim + 1) * 8  # with label
    total_bytes = file_size_gb * (10**9)
    n = total_bytes // bytes_per_point  # the number of require n
    return dim, k, int(n)

# ------------------------create big data-----------------
# dim1, k1, n1 = calculate_n()
# part1.generate_data(dim1, k1, n1, out_path="dataForBfr.csv", points_gen=part1.creatingPoints)
# dim2, k2, n2 = calculate_n()
# part1.generate_data(dim2, k2, n2, out_path="dataForCure.csv", points_gen=part1.creatingPoints)


import math

def euclideanDistance(point1, point2):
    """ Calculates the euclidean distance between two points. """
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

def calculate_davies_bouldin_index(file_path, block_size=1000):
    """ Calculates the Davies-Bouldin index for a large file. """

    cluster_sums = {}  # Sum of points for each cluster
    cluster_counts = {}  # Number of points in each cluster
    cluster_squares_sums = {} # Sum of squares of distances

    with open(file_path, 'r') as file:
        block = []
        for line in file:
            block.append(line.strip().split(','))
            if len(block) >= block_size:
                cluster_sums, cluster_counts, cluster_squares_sums = process_block(block, cluster_sums, cluster_counts, cluster_squares_sums)
                block = []

        # Process the remaining block
        if block:
            cluster_sums, cluster_counts, cluster_squares_sums = process_block(block, cluster_sums, cluster_counts, cluster_squares_sums)

    # Calculate final centroids
    centroids = {cluster: [s / cluster_counts[cluster] for s in cluster_sums[cluster]] for cluster in cluster_sums}

    #calculate scatters
    scatters = {}
    for cluster in cluster_sums:
        scatters[cluster] = np.sqrt(cluster_squares_sums[cluster]/cluster_counts[cluster])

    #calculate similarity matrix
    similarity_matrix = np.zeros((len(centroids),len(centroids)))
    for i,cluster1 in enumerate(centroids):
        for j,cluster2 in enumerate(centroids):
            if i != j:
                similarity_matrix[i][j] = (scatters[cluster1] + scatters[cluster2])/euclideanDistance(centroids[cluster1],centroids[cluster2])

    #calculate davies bouldin index
    davies_bouldin_index = np.mean(np.max(similarity_matrix,axis=1))

    return davies_bouldin_index

def process_block(block, cluster_sums, cluster_counts, cluster_squares_sums):
    """ Processes a block of data and updates the metrics. """

    for row in block:
        point = np.array(row[:-1], dtype=float)
        cluster = int(row[-1])

        # Update cluster sums and counts
        if cluster not in cluster_sums:
            cluster_sums[cluster] = np.zeros_like(point)
            cluster_counts[cluster] = 0
            cluster_squares_sums[cluster] = 0
        cluster_sums[cluster] += point
        cluster_counts[cluster] += 1
        cluster_squares_sums[cluster] += np.sum(np.square(point - cluster_sums[cluster]/cluster_counts[cluster]))

    return cluster_sums, cluster_counts, cluster_squares_sums

# Example usage
# file_path = 'dataForBfr.csv'  # Replace with your file path
dim = random.randint(5, 20)
k = random.randint(5, 20)
Part2.bfr_cluster(dim, k, 1000, 1000,'dataForBfr.csv' , 'resultDataForBfr.csv')
davies_bouldin_index_bfr = calculate_davies_bouldin_index('resultDataForBfr.csv')
print(f'Davies-Bouldin index for bfr: {davies_bouldin_index_bfr}')

Part2.cure_cluster(dim, k, 4000, 1000,'dataForCure.csv' , 'resultDataForCure.csv')
davies_bouldin_index_cure = calculate_davies_bouldin_index('resultDataForCure.csv')
print(f'Davies-Bouldin index for cure: {davies_bouldin_index_cure}')