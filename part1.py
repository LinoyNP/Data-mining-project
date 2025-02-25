import csv
import random
import numpy as np


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


generate_data(3, 4, 50, "out_path.csv", points_gen=creatingPoints, extras = {})