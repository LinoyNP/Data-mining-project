#Name: Linoy Nisim Pur  ID:324029685
#Name: Noa Shem Tov     ID:207000134
"-------------------------------------------------part 2 of 3-----------------------------------------------------------"
import csv
import math
import os
import random
import part1 # Assuming part1 contains required functions like load_points and h_clustering
import numpy as np


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

"-----------------------------------------------------------------------------------------------------------------------"

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
    std_dev = np.sqrt(np.maximum(variance, 1))#  1e-10 to prevent Divide by zero
    return centroid, std_dev

def update_cluster_file(input_file, old_cluster_id, new_cluster_id,KindOfCluster,KindOfClusterDestination):
    """
       Updates cluster IDs and cluster types in a CSV file.

       This function reads a CSV file, searches for rows where the cluster type matches `KindOfCluster`
       and the cluster ID matches `old_cluster_id`, and updates them to `new_cluster_id` and `KindOfClusterDestination`.

       The update is performed in a temporary file, which then replaces the original file to ensure atomic modification.

       Parameters:
       - input_file (str): The path to the CSV file.
       - old_cluster_id (int or str): The cluster ID to be replaced.
       - new_cluster_id (int or str): The new cluster ID to assign.
       - KindOfCluster (str): The cluster type to be updated.
       - KindOfClusterDestination (str): The new cluster type to assign.

       Returns:
       - str: The path of the updated CSV file.

       Behavior:
       - Reads the file line by line.
       - Extracts the data, assuming the last two columns represent the cluster ID and cluster type.
       - Updates only the matching cluster IDs and types.
       - Writes the modified data to a temporary file and then replaces the original file.

       """
    with open(input_file, 'r') as infile, open(input_file + '.tmp', 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        for line in infile:
            data = line.strip().split(',')
            point, cluster_id, cluster_type = data[:-1], data[-2], data[-1]
            if cluster_type == KindOfCluster and cluster_id == str(old_cluster_id):
                cluster_id = new_cluster_id
                cluster_type = KindOfClusterDestination

            writer.writerow(point + [cluster_id] + [cluster_type])

    os.remove(input_file)
    os.rename(input_file + '.tmp', input_file)
    return input_file


def merge_clusters(cluster_dict, threshold, input_file,KindOfClusterSource,KindOfClusterDestination,counterOfClusterSource):
    """
       Merges clusters if the Mahalanobis distance between their centroids is less than the specified threshold.

       This function iterates over all cluster pairs in `cluster_dict` and calculates the Mahalanobis distance
       between their centroids. If the distance is below `threshold`, the clusters are merged, and their
       information is updated in the input file.

       Parameters:
       - cluster_dict (dict): A dictionary where keys are cluster IDs, and values are tuples (N, SUM, SUMSQ) representing:
           - N: Number of points in the cluster.
           - SUM: Sum of all points in the cluster.
           - SUMSQ: Sum of squares of all points in the cluster.
       - threshold (float): The distance threshold for merging clusters.
       - input_file (str): The path to the CSV file that contains cluster assignments. It will be updated accordingly.
       - KindOfClusterSource (str): The original type of the clusters before merging.
       - KindOfClusterDestination (str): The new type of the clusters after merging.
       - counterOfClusterSource (int): A counter tracking the number of clusters before and after merging.

       Returns:
       - dict: A dictionary containing the updated merged clusters, with new cluster IDs.
       - str: The updated input file path after modifying cluster assignments.
       - int: The updated cluster count after merging.

       Behavior:
       - Iterates through the clusters, comparing centroids using the Mahalanobis distance.
       - If two clusters are close enough (below `threshold`), they are merged.
       - The `update_cluster_file` function is called to reflect the changes in the CSV file.
       - The `UnionCluster` function updates the cluster parameters (N, SUM, SUMSQ).
       - Returns the final merged cluster dictionary, updated file path, and updated cluster count.
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
                input_file = update_cluster_file(input_file, keys[j], keys[i],KindOfClusterSource,KindOfClusterDestination)
                N1, SUM1, SUMSQ1 = UnionCluster(N1, SUM1, SUMSQ1, N2, SUM2, SUMSQ2)
                merged_ids.add(keys[j])
                counterOfClusterSource -= 1
        merged[new_id] = (N1, SUM1, SUMSQ1)
        merged_ids.add(keys[i])
        new_id += 1

    return merged , input_file , counterOfClusterSource

def MergePointToCluster(point,N,SUM,SUMSQ):
    """
      Merges a new data point into an existing cluster by updating the cluster's statistical properties.

      This function updates the cluster parameters (N, SUM, SUMSQ) to reflect the addition of a new data point.

      Parameters:
      - point (array-like): The new data point to be added to the cluster.
      - N (int): The current number of points in the cluster.
      - SUM (numpy array): The sum of all points in the cluster.
      - SUMSQ (numpy array): The sum of squared values of all points in the cluster.

      Returns:
      - int: The updated number of points in the cluster (newN).
      - numpy array: The updated sum of points in the cluster (newSum).
      - numpy array: The updated sum of squared points in the cluster (newSUMSQ).

      Behavior:
      - Converts the input point into a NumPy array for efficient computation.
      - Increments the cluster's count (`N`).
      - Updates the sum (`SUM`) by adding the new point's values.
      - Updates the sum of squares (`SUMSQ`) by adding the squared values of the new point.
"""
    arrayPoint = np.array(point)
    newN = 1 + N
    newSum = arrayPoint + SUM
    newSUMSQ = SUMSQ + arrayPoint**2
    return newN, newSum, newSUMSQ

def load_points(in_path, dim, X, start_row, points=[]):
    """
    Reads points from a CSV file in stages.
    :param in_path: The input file path.
    :param dim: The number of dimensions for each data point.
    :param X: The number of rows to read at a time.
    :param start_row: The row to start reading from (used for incremental reads).
    :param points: A list to append the points to.
    """
    with open(in_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Move to the start_row position by iterating through the rows
        for _ in range(start_row):
            next(reader)
        # Read up to X rows or until the end of the file
        counter = 0
        for row in reader:
            if counter >= X:
                break
            points.append([float(x) for x in row[:dim]])
            counter += 1
        # Return the next row to start from for the next call
        return points, counter + start_row  # the next start_row will be this


def FinalFunctionBFR(input_file, dim):
    """
        Processes a CSV file by keeping only the first 'dim' columns,
        the second-to-last column, and the last column. The function
        modifies the original file in place.

        Parameters:
        - input_file (str): The path to the input CSV file.
        - dim (int): The number of dimensions to retain from the beginning of each row.

        Returns:
        - str: The path of the updated file.

        Behavior:
        - Reads the input CSV file line by line.
        - Extracts the first 'dim' columns, the second-to-last column, and the last column.
        - Writes the selected columns into a temporary file.
        - Replaces the original file with the temporary file to keep the changes.
        """
    with open(input_file, 'r') as infile, open(input_file + '.tmp', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for line in infile:
            data = line.strip().split(',')
            first_columns = data[:dim]
            second_last_column = data[-2]
            last_column = data[-1]
            writer.writerow(first_columns + [second_last_column, last_column])
    os.remove(input_file)
    os.rename(input_file + '.tmp', input_file)
    return input_file

def bfr_cluster(dim, k, n, block_size, in_path, out_path):
    """
    Implements the BFR (Bradley-Fayyad-Reina) clustering algorithm to efficiently cluster
    large datasets using a combination of summarization and iterative clustering.

    Parameters:
    -----------
    dim : int
        The number of dimensions (features) in the dataset.
    k : int or None
        The number of clusters. If None, the optimal k is determined dynamically.
    n : int
        The total number of data points in the dataset.
    block_size : int
        The number of points to process in each iteration (block-wise processing).
    in_path : str
        The path to the input CSV file containing the data points.
    out_path : str
        The path where the output CSV file with clustering results will be saved.

    Description:
    ------------
    The BFR algorithm maintains three key sets of points:
      - **Discard Set (DS)**: Points tightly clustered around a centroid (maintained as summary statistics).
      - **Compressed Set (CS)**: Intermediate clusters that are not part of DS but have multiple points.
      - **Retained Set (RS)**: Points that are currently unclustered (outliers or potential future clusters).

    Algorithm Steps:
    ---------------
    1. **Initialize Clustering**:
       - Load the first `block_size` points and determine `k` if not provided.
       - Perform initial clustering using k-means and store clusters in the **Discard Set (DS)**.

    2. **Iterate Over Remaining Blocks**:
       - Load a new batch of `block_size` points.
       - Assign points to DS clusters if they are within the Mahalanobis distance threshold.
       - If a point does not fit in DS, attempt to assign it to CS.
       - If no cluster fits, add the point to RS.
       - If RS accumulates enough points, attempt k-means to find new clusters for CS.
       - Merge similar CS clusters.

    3. **Final Merging**:
       - Merge CS clusters into DS if they are close enough.
       - Perform a final clustering pass on any remaining points.
       - Ensure the final number of DS clusters is at most `k`.

    4. **Output the Clustered Data**:
       - Save the final clustering results in the output CSV file.
       - Perform file cleanup using `FinalFunctionBFR()` to organize the output file.

    Notes:
    ------
    - The function uses **Mahalanobis distance** to measure closeness between points and cluster centroids.
    - Cluster summaries are maintained using `(N, SUM, SUMSQ)`, which allows efficient updates.
    - Compressed and Discard clusters are periodically merged to optimize the number of clusters.

    Returns:
    --------
    None
        The function processes and updates the output file at `out_path` with the cluster assignments.
    """
    DS = {}  # {cluster_id: (N, SUM, SUMSQ)}
    CS = {}  # {cluster_id: (N, SUM, SUMSQ)}
    RS = []# Retained Set: Points that were not assigned to any cluster
    DSstr = "DS"
    CSstr = "CS"
    clusterIDCounterDS = 0
    clusterIDCounterCS = 0
    threshold = 2 #after we search about this
    start_row = 0 # represent the number of the row in the file
    # Load first block
    AllPoints = []
    AllPoints, start_row = load_points(in_path, dim, block_size, start_row, AllPoints)
    if k is None:
        k = part1.FindKOptimal(dim,block_size//10,AllPoints, 100)

    AllPoints = [tuple(sublist) for sublist in AllPoints]
    initialClusters = part1.k_means(dim, k, block_size, AllPoints)
    #in the first iteration we just put all the points un the first block_size in the DS
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for cluster in initialClusters:
            N, SUM, SUMSQ = RepresentClusterAsVector(np.array(cluster))
            DS[clusterIDCounterDS] = (N, SUM, SUMSQ)
            for point in cluster:
                writer.writerow(list(point) + [clusterIDCounterDS] + [DSstr])
            clusterIDCounterDS += 1

    total_blocks = (n / block_size) - 1
    numberOfIterations = math.ceil(total_blocks)# Rounds up to ensure full coverage if there's a remainder
    for _ in range(numberOfIterations):
        block_points = []
        block_points, start_row = load_points(in_path, dim, block_size, start_row, block_points)
        for point in block_points:
            assigned = False
            for cluster_id, (N, SUM, SUMSQ) in DS.items():
                centroid, std = computeCentroidPerClusterAndStd(N, SUM, SUMSQ)
                if Mahalanobis(np.array(point), centroid, std) < threshold:
                    DS[cluster_id] = MergePointToCluster(point, N, SUM, SUMSQ)
                    assigned = True
                    with open(out_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(list(point) + [clusterIDCounterDS] + [DSstr])
                        break
            if not assigned:
                assigned = False
                if clusterIDCounterCS:
                    for cluster_id, (N, SUM, SUMSQ) in CS.items():
                        centroid, std = computeCentroidPerClusterAndStd(N, SUM, SUMSQ)
                        if Mahalanobis(np.array(point), centroid, std) < threshold:
                            CS[cluster_id] = MergePointToCluster(point, N, SUM, SUMSQ)
                            assigned = True
                            with open(out_path, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow(list(point) + [clusterIDCounterCS] + [CSstr])
                            break
                if not assigned:
                    RS.append(point)

        # If RS contains at least 2 * k points, attempt to find new clusters
        # his ensures that we only run K-Means on RS when we have enough data
        if len(RS) >= 2 * k:
            newRS = RS[:]
            RS_clusters, RS_centroids = part1.run_k_means(dim, k, len(RS), RS, 100)
            for cluster in RS_clusters:
                #only if the cluster has more than 1 point
                if len(cluster) > 1:
                    # If a valid cluster is found, move it to CS (Compressed Set)
                    N, SUM, SUMSQ = RepresentClusterAsVector(np.array(cluster))
                    CS[clusterIDCounterCS] = (N, SUM, SUMSQ)
                    for point in cluster:
                        with open(out_path, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(list(point) + [clusterIDCounterCS] + [CSstr])
                        newRS.remove(point)
                    clusterIDCounterCS += 1
            RS = newRS[:]
        # Merge close CS clusters based on threshold
        CS , out_path, clusterIDCounterCS= merge_clusters(CS, threshold,out_path,"CS","CS",clusterIDCounterCS)

    # Final pass: Merge remaining RS clusters into CS
    if len(RS) > 0:
        RS_clusters, RS_centroids = part1.run_k_means(dim, k, len(RS), RS, 100)
        for i, cluster in enumerate(RS_clusters):
            if len(cluster) > 1:
                N, SUM, SUMSQ = RepresentClusterAsVector(np.array(cluster))
                CS[clusterIDCounterCS] = (N, SUM, SUMSQ)
                for point in cluster:
                    with open(out_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(list(point) + [clusterIDCounterCS] + [CSstr])
                clusterIDCounterDS += 1
    # Merge CS clusters into DS if they are close enough
    for cs_id, (N_cs, SUM_cs, SUMSQ_cs) in CS.items():
        cs_centroid, cs_std = computeCentroidPerClusterAndStd(N_cs, SUM_cs, SUMSQ_cs)
        merged = False
        for ds_id, (N_ds, SUM_ds, SUMSQ_ds) in DS.items():
            ds_centroid, ds_std = computeCentroidPerClusterAndStd(N_ds, SUM_ds, SUMSQ_ds)
            if Mahalanobis(cs_centroid, ds_centroid, ds_std) < threshold:
                out_path = update_cluster_file(out_path, cs_id, ds_id, CSstr,DSstr)
                DS[ds_id] = UnionCluster(N_ds, SUM_ds, SUMSQ_ds, N_cs, SUM_cs, SUMSQ_cs)
                merged = True
                break
        if not merged:
            out_path = update_cluster_file(out_path, cs_id, clusterIDCounterDS, CSstr, DSstr)
            DS[clusterIDCounterDS] = (N_cs, SUM_cs, SUMSQ_cs)
            clusterIDCounterDS += 1
    # Final clustering assignment
    while(clusterIDCounterDS > k):
        DS ,out_path,clusterIDCounterDS= merge_clusters(DS, threshold,out_path,"DS","DS",clusterIDCounterDS)
    FinalFunctionBFR(out_path,dim)# orgenize the file

bfr_cluster(3, None, 1000, 200, "out_path.csv", "blabla.csv")

"-----------------------------------------------------------------------------------------------------------------------"

def cure_cluster(dim, k, n, block_size, in_path, out_path):
    """
       Implements the CURE clustering algorithm.
       :param dim: Dimension of the data points.
       :param k: Number of clusters.
       :param n: Total number of data points.
       :param block_size: Number of points loaded per block.
       :param in_path: Input file path containing data points.
       :param out_path: Output file path to save clustered points.
    """
    #Step 1
    # Load the first block of points
    AllPoints = []
    start_row = 0
    AllPoints, start_row = load_points(in_path, dim, block_size, start_row, AllPoints)
    AllPoints = [tuple(sublist) for sublist in AllPoints] # Convert lists to tuples for immutability

    #Perform hierarchical clustering to get k initial clusters
    clustersForStepOne = []
    clustersForStepOne = part1.h_clustering(dim, k, AllPoints,None, clustersForStepOne)

    # Define the number of representative points per cluster
    # We use math.ceil(n / block_size) to ensure that the loop runs enough iterations
    # so that all points are processed, even if n is not perfectly divisible by block_size.
    # When n is not a multiple of block_size, ceil ensures one extra iteration
    # to process the remaining points.
    points_per_cluster = max(k,block_size) // min(k,block_size) #the amount of representative points for each cluster

    # Compute centroids for each cluster
    Centroids = []
    for cluster in clustersForStepOne:
        cluster_array = np.array(cluster)
        centroid = np.sum(cluster_array, axis=0) / len(cluster_array) # =0 for sum by columns
        Centroids.append(tuple(map(float, centroid)))
    #Finding the point that are dispersed as possible
    # Select representative points (most dispersed from centroid)
    RepresentativePoints = []
    for i, cluster in enumerate(clustersForStepOne):
        listOfDistancBetweenPointInCurrentCluster = []
        for point in cluster:
            distance = part1.euclideanDistance(point, Centroids[i])
            listOfDistancBetweenPointInCurrentCluster.append((distance, point))
        listOfDistancBetweenPointInCurrentCluster.sort(key=lambda x: x[0], reverse=True)
        farthest_points = [point for _, point in listOfDistancBetweenPointInCurrentCluster[:points_per_cluster]]
        RepresentativePoints.append(farthest_points)
    # Compute new representative points (midpoint between the representative point and centroid)
    # Adjust representative points by shrinking towards centroid
    UpdatedRepresentativePoints = []
    for i, cluster_representatives in enumerate(RepresentativePoints):
        centroid = np.array(Centroids[i])
        updated_points = []
        for rep_point in cluster_representatives:
            # First midpoint: between representative point and centroid
            midpoint1 = (np.array(rep_point) + centroid) / 2
            # Second midpoint: between the first midpoint and the representative point
            midpoint2 = (midpoint1 + np.array(rep_point)) / 2
            # Append the final updated representative point
            updated_points.append(midpoint2)
        UpdatedRepresentativePoints.append(updated_points)

    # Step 2: Assign each point to the closest cluster based on the representative points
    final_assignments = [[] for _ in range(k)]  #  List to store assigned points per cluster
    NumOfIteartion = math.ceil(n / block_size)
    start_row = 0
    for _ in range(NumOfIteartion):
        AllPoints = []
        AllPoints, start_row = load_points(in_path, dim, block_size, start_row, AllPoints)
        for point in AllPoints:
            min_distance = float('inf')
            closest_cluster = -1

            # Calculate the distance from the point to all representative points
            # Find the closest representative point
            for cluster_idx, rep_points in enumerate(UpdatedRepresentativePoints):
                for rep_point in rep_points:
                    distance = part1.euclideanDistance(point, rep_point)
                    if distance < min_distance:
                        min_distance = distance
                        closest_cluster = cluster_idx

            # Assign the point to the closest cluster
            final_assignments[closest_cluster].append(point)

    # Write the final assignments to a CSV file
    # Save the final assignments to a CSV file
    with open(out_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for cluster_idx, cluster in enumerate(final_assignments):
            for point in cluster:
                writer.writerow(list(point) + [cluster_idx])
## Example usage
#cure_cluster(3, 2, 4660, 3, "out_path.csv", "noa")