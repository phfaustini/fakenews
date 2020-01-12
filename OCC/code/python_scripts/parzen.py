#https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
# python python_scripts/parzen.py
from sys import argv

from sklearn.neighbors.kde import KernelDensity
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATASETS_FALSE = [
    "datasets/Websites/btv-lifestyle/Structured/features_False.csv",                 # 0
    "datasets/WhatsApp/whats_br/Structured/features_False.csv",                      # 1
    "datasets/Websites/FakeBrCorpus/Structured/features_False.csv",                  # 2
    "datasets/Twitter/tweets_br/Structured/features_False.csv",                      # 3
    "datasets/Websites/fakenewsdata1_randomPolitics/Structured/features_False.csv",  # 4
    "datasets/Websites/Bhattacharjee/Structured/features_False.csv"                  # 5
]

DATASETS_TRUE = [
    "datasets/Websites/btv-lifestyle/Structured/features_True.csv",                  # 0
    "datasets/WhatsApp/whats_br/Structured/features_True.csv",                       # 1
    "datasets/Websites/FakeBrCorpus/Structured/features_True.csv",                   # 2
    "datasets/Twitter/tweets_br/Structured/features_True.csv",                       # 3
    "datasets/Websites/fakenewsdata1_randomPolitics/Structured/features_True.csv",   # 4
    "datasets/Websites/Bhattacharjee/Structured/features_True.csv"                   # 5
]

FEATURES = ["uppercase", "exclamation", "has_exclamation", "question", "adj", "adv", "noun", "spell_errors", "lexical_size", "polarity", "number_sentences", "len_text", "words_per_sentence", "word2vec", "label"]

def load_false(dataset: int):
    X = np.genfromtxt(DATASETS_FALSE[dataset], delimiter=',', encoding='utf8')
    return pd.DataFrame(X, columns=FEATURES)
def load_true(dataset: int):
    X = np.genfromtxt(DATASETS_TRUE[dataset], delimiter=',', encoding='utf8')
    return pd.DataFrame(X, columns=FEATURES)

def parzen(dataset: int, feature: str) -> None:
    b = 0.1
    X = load_false(dataset)
    scaler = StandardScaler(copy=False)
    X_transformed = scaler.fit_transform(X)
    kdex = KernelDensity(kernel='gaussian', bandwidth=b)
    fx = X[feature].values.reshape((-1, 1))
    kdex.fit(fx)
    x_d = np.linspace(min(fx.flatten())-.5, max(fx.flatten())+.5, 1000)
    logprob = kdex.score_samples(x_d[:,None])

    Y = load_true(dataset)
    Y_transformed = scaler.transform(Y)
    kdey = KernelDensity(kernel='gaussian', bandwidth=b)
    fy = Y[feature].values.reshape((-1, 1))
    kdey.fit(fy)
    y_d = np.linspace(min(fy.flatten())-.5, max(fy.flatten())+.5, 1000)
    logproby = kdey.score_samples(y_d[:,None])
    
    plt.clf()
    plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
    plt.plot(fx, np.full_like(fx, -0.01), '|k', label='Inliers', markeredgewidth=1)
    plt.fill_between(y_d, np.exp(logproby), alpha=0.5)
    plt.plot(fy, np.full_like(fy, -0.01), '.k', label='Outliers' ,markeredgewidth=1)

    d = DATASETS_FALSE[dataset].split("/")[2]
    #plt.title(feature)
    plt.legend(loc='best')
    plt.ylim(-0.02, 1.1)
    #plt.show()
    plt.savefig("results/plots_features/"+d+"/"+feature+'.svg')


if __name__ == "__main__":
    #i = int(argv[1])
    #f = argv[2]
    #parzen(i, f)
    for i in [0,1,2,3,4,5]: # datasets
        for f in FEATURES[:-1]:
            parzen(i, f)
