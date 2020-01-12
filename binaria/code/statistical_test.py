import numpy as np
import Orange
import matplotlib.pyplot as plt

data = np.array([
    [.5, .69, .62, .85],
    [.71, .76, .8, .73],
    [.73, .84, .81, .91],
    [.74, .77, .8, .88],
    
    [.33, .7, .54, .69],
    [.73, .77, .8, .64],
    [.79, .78, .56, .83],
    [.78, .77, .66, .86],

    [.4, .68, .63, .76],
    [.67, .82, .85, .79],
    [.74, .89, .78, .94],
    [.76, .84, .84, .90],

    [.78, .91, .68, .91],
    [.8, .9, .92, .94],
    [.83, .94, .72, .93],
    [.83, .93, .82, .95],

    [.49, .45, .71, .73],
    [.63, .67, .79, .77],
    [.60, .69, .79, .0],
    [.62, .71, .77, .64],

    [.59, .51, .48, .54],
    [.43, .55, .43, .59],
    [.47, .76, .48, .0],
    [.44, .68, .46, .48]])

ranks = np.array([
    [4, 2, 3, 1],
    [4,2,1,3],
    [4,2,3,1],
    [4,3,2,1],

    [4, 1, 3, 2],
    [3,2,1,4],
    [2,3,4,1],
    [2,3,4,1],

    [4,2,3,1],
    [4,2,1,3],
    [4,2.5,2.5,1],
    [4, 2.5, 2.5, 1],

    [3, 1.5, 4, 1.5],
    [4,3,2,1],
    [3,1,4,2],
    [3,2,4,1],

    [3,4,2,1],
    [4,3,1,2],
    [3,2,1,4],
    [4,2,1,3],

    [1,3,4,2],
    [3.5,2,3.5,1],
    [3,1,2,4],
    [4,1,3,2]
    ])

average_rank = np.sum(ranks,axis=0) / ranks.shape[0]


def chi2f(N: int, k: int, ranks: np.ndarray) -> float:
    a = ( (12*N) / ( k*(k+1) ))
    b = 0
    for r in ranks:
        b += r**2
    b -= ( ( k* (k+1)**2 ) /4)
    return a*b

def Ff(N: int, k: int, x: float) -> float:
    return ( (N-1)*x )   /   ( N*(k-1) - x )

def CD(alpha: float, N: int, k: int) -> float:
    return alpha * np.sqrt( (k*(k+1)) / (6*N) )

x = chi2f(data.shape[0], data.shape[1], average_rank)
t = Ff(data.shape[0], data.shape[1], x)

f1 = data.shape[1] - 1
f2 = f1 * (data.shape[0] - 1)

cd = CD(2.569, data.shape[0], data.shape[1])


def distances(average_rank: np.ndarray) -> np.ndarray:
    n = average_rank.shape[0]
    dst = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dst[i, j] = average_rank[i] - average_rank[j]
    return dst

print("N = {0}".format(data.shape[0]))
print("k = {0}".format(data.shape[1]))
print("x = {0}".format(x))
print("Ff = {0}".format(t))
print("F({0},{1})".format(f1, f2))
print("cd (I found) = {0}".format(cd))
print("Average ranks = {0}".format(average_rank))
print("Distances:")
print(distances(average_rank))

names = ["Custom features", "Word2Vec", "DCDistance", "BOW" ]
cd = Orange.evaluation.compute_CD(average_rank, data.shape[0])
print("cd (Library found) = {0}".format(cd))
Orange.evaluation.graph_ranks(average_rank, names, cd=cd, width=6, textspace=1.5, filename="results/critical_differences.pdf")  # https://docs.biolab.si//3/data-mining-library/reference/evaluation.cd.html
