import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from fakenews_detector.model import Model

language = 'pt'
dataset = 'FakeBrCorpus'
platform = 'Websites'
tree_estimators = np.arange(1, 1001, 50)

t = Model(language='pt', embeddings_path='embeddings/pt/model.txt', platform_folder='datasets/Websites/FakeBrCorpus/')


# FEATURES
print("FEATURES + GAUSSIANNB")
t.classify(dataset=dataset, platform=platform, language=language, clf=GaussianNB(), parameters={}, feature_set=Model.FEATURES_CODE, kfold=5)
print("------------------------------------------------------")

print("FEATURES + KNN")
t.classify(dataset=dataset, platform=platform, language=language, clf=KNeighborsClassifier(), parameters={'clf__n_neighbors':[1,3,5,7]}, feature_set=Model.FEATURES_CODE, kfold=5)
print("------------------------------------------------------")

print("FEATURES + SVC")
t.classify(dataset=dataset, platform=platform, language=language, clf=SVC(random_state=42), parameters={'clf__kernel':['poly', 'rbf', 'linear', 'sigmoid'], 'clf__gamma':['auto']}, feature_set=Model.FEATURES_CODE, kfold=5)
print("------------------------------------------------------")

print("FEATURES + RF")
t.classify(dataset=dataset, platform=platform, language=language, clf=RandomForestClassifier(random_state=42), parameters={'clf__n_estimators':tree_estimators}, feature_set=Model.FEATURES_CODE, kfold=5)
print("------------------------------------------------------")


# WORD2VEC
print("WORD2VEC + GAUSSIANNB")
t.classify(dataset=dataset, platform=platform, language=language, clf=GaussianNB(), parameters={}, feature_set=Model.WORD2VEC_CODE, kfold=5)
print("------------------------------------------------------")

print("WORD2VEC + KNN")
t.classify(dataset=dataset, platform=platform, language=language, clf=KNeighborsClassifier(), parameters={'clf__n_neighbors':[1,3,5,7]}, feature_set=Model.WORD2VEC_CODE, kfold=5)
print("------------------------------------------------------")

print("WORD2VEC + SVC")
t.classify(dataset=dataset, platform=platform, language=language, clf=SVC(random_state=42), parameters={'clf__kernel':['poly', 'rbf', 'linear', 'sigmoid'], 'clf__gamma':['auto']}, feature_set=Model.WORD2VEC_CODE, kfold=5)
print("------------------------------------------------------")

print("WORD2VEC + RF")
t.classify(dataset=dataset, platform=platform, language=language, clf=RandomForestClassifier(random_state=42), parameters={'clf__n_estimators':tree_estimators}, feature_set=Model.WORD2VEC_CODE, kfold=5)
print("------------------------------------------------------")


# DCDISTANCE
print("DCDISTANCE + GAUSSIANNB")
t.classify(dataset=dataset, platform=platform, language=language, clf=GaussianNB(), parameters={}, feature_set=Model.DCDISTANCE_CODE, kfold=5)
print("------------------------------------------------------")

print("DCDISTANCE + KNN")
t.classify(dataset=dataset, platform=platform, language=language, clf=KNeighborsClassifier(), parameters={'clf__n_neighbors':[1,3,5,7]}, feature_set=Model.DCDISTANCE_CODE, kfold=5)
print("------------------------------------------------------")

print("DCDISTANCE + SVC")
t.classify(dataset=dataset, platform=platform, language=language, clf=SVC(random_state=42), parameters={'clf__kernel':['poly', 'rbf', 'linear', 'sigmoid'], 'clf__gamma':['auto']}, feature_set=Model.DCDISTANCE_CODE, kfold=5)
print("------------------------------------------------------")

print("DCDISTANCE + RF")
t.classify(dataset=dataset, platform=platform, language=language, clf=RandomForestClassifier(random_state=42), parameters={'clf__n_estimators':tree_estimators}, feature_set=Model.DCDISTANCE_CODE, kfold=5)
print("------------------------------------------------------")


# BOW
print("BOW + MultinomialNB")
t.classify(dataset=dataset, platform=platform, language=language, clf=MultinomialNB(), parameters={}, feature_set=Model.BOW_CODE, kfold=5)
print("------------------------------------------------------")

print("BOW + KNN")
t.classify(dataset=dataset, platform=platform, language=language, clf=KNeighborsClassifier(), parameters={'clf__n_neighbors':[1,3,5,7]}, feature_set=Model.BOW_CODE, kfold=5)
print("------------------------------------------------------")

print("BOW + SVC")
t.classify(dataset=dataset, platform=platform, language=language, clf=SVC(random_state=42), parameters={'clf__kernel':['poly', 'rbf', 'linear', 'sigmoid'], 'clf__gamma':['auto']}, feature_set=Model.BOW_CODE, kfold=5)
print("------------------------------------------------------")

print("BOW + RF")
t.classify(dataset=dataset, platform=platform, language=language, clf=RandomForestClassifier(random_state=42), parameters={'clf__n_estimators':tree_estimators}, feature_set=Model.BOW_CODE, kfold=5)
print("------------------------------------------------------")
