# python -W ignore sklearn/optics_exploratory.py
from sys import argv

from sklearn.cluster import OPTICS  # pip install scikit-learn==0.21.3 # No with 0.20.3
from sklearn.preprocessing import StandardScaler
import numpy as np

DATASETS_FALSE = [
    "datasets/Websites/btv-lifestyle/Structured/features_False.csv",                 # 0
    "datasets/WhatsApp/whats_br/Structured/features_False.csv",                      # 1
    "datasets/Websites/FakeBrCorpus/Structured/features_False.csv",                  # 2
    "datasets/Twitter/tweets_br/Structured/features_False.csv",                      # 3
    "datasets/Websites/fakenewsdata1_randomPolitics/Structured/features_False.csv",  # 4
    "datasets/Websites/Bhattacharjee/Structured/features_False.csv"                  # 5
]


def exploratory_analysis(dataset: str, samples=0.1, eps=np.inf) -> None:
    X = np.genfromtxt(dataset, delimiter=',', encoding='utf8')
    scaler = StandardScaler(copy=False)
    X_transformed = scaler.fit_transform(X)
    clust = OPTICS(min_samples=samples, max_eps=eps, n_jobs=2)
    labels = clust.fit_predict(X)
    n_clusters = len(set(labels))
    print("# clusters: {0}".format(n_clusters))
    #for l in labels:
    #    print(l, end=" ")

def search() -> None:
    samples = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    eps = [1, 1.3, 1.6, 2, 2.3, 2.6, 3]
    for dataset in DATASETS_FALSE:
        print("\n\nDataset {0}".format(dataset))
        for s in samples:
            for e in eps:
                print("min_samples = {0}, max_eps = {1} with".format(s, e), end=" ")
                exploratory_analysis(dataset, s, e)

if __name__ == "__main__":
    #i = int(argv[1])
    #exploratory_analysis(DATASETS_FALSE[i])
    search()
