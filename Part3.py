#Name: Linoy Nisim Pur  ID:324029685
#Name: Noa Shem Tov     ID:207000134
"-------------------------------------------------part 3 of 3-----------------------------------------------------------"
import random
import part1

def CreateSmallDate():
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

    :return: None
    """
    currentN = random.randint(1000, 3000)
    currentK = random.randint(5,currentN // 10)
    part1.generate_data(3, currentK, currentN, "smallDataPath.csv", points_gen=part1.creatingPoints, extras={})


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
    dim1, k1 = generateValid_K1AndDim()
    n = random.randint(k1*100, k1*500)
    part1.generate_data(dim1, k1, n, dataPath, points_gen=part1.creatingPoints, extras={})

CreateDateWith_K_clustersAnd_Dim("dataOneOfPartOne.csv")
CreateDateWith_K_clustersAnd_Dim("dataTwoOfPartOne.csv")
CreateDateWith_K_clustersAnd_Dim("dataThreeOfPartOne.csv")