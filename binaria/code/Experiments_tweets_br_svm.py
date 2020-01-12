import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from fakenews_detector.model import Model

dataset = "tweets_br"
platform = 'Twitter'
language = "pt"
tree_estimators = np.arange(1, 1001, 50)

t = Model(language='pt', embeddings_path='embeddings/pt/model.txt', platform_folder='datasets/Twitter/tweets_br/')


##########################################################################################################
print("TESTING WITH LTO")
print()
print()

# FEATURES


print("BOW + SVC")
fscore_topic_svm,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', SVC(kernel='poly'), Model.BOW_CODE)
print("------------------------------------------------------")


