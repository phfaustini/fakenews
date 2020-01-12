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
print("FEATURES + GAUSSIANNB")
t.custom_gridsearch(dataset=dataset, platform=platform, parameters={}, language=language, clf="gnb", feature_set=Model.FEATURES_CODE)
fscore_topic_nb,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', GaussianNB(), Model.FEATURES_CODE)
pd.Series(fscore_topic_nb).to_pickle('results/f1scores_twitterLTO_features_nb.pkl')
print("------------------------------------------------------")

print("FEATURES + KNN")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__n_neighbors':[1,3,5,7]}, language=language, clf="knn", feature_set=Model.FEATURES_CODE)
knn_k_features = params['clf__n_neighbors']
fscore_topic_knn,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', KNeighborsClassifier(n_neighbors=knn_k_features), Model.FEATURES_CODE)
pd.Series(fscore_topic_knn).to_pickle('results/f1scores_twitterLTO_features_knn.pkl')
print("------------------------------------------------------")

print("FEATURES + SVC")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__kernel':['poly', 'rbf', 'linear', 'sigmoid']}, language=language, clf="svm", feature_set=Model.FEATURES_CODE)
svc_kernel_features = params['clf__kernel']
fscore_topic_svm,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', SVC(kernel=svc_kernel_features), Model.FEATURES_CODE)
pd.Series(fscore_topic_svm).to_pickle('results/f1scores_twitterLTO_features_svm.pkl')
print("------------------------------------------------------")

print("FEATURES + RF")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__n_estimators':tree_estimators}, language=language, clf="rf", feature_set=Model.FEATURES_CODE)
rf_est_features = params['clf__n_estimators']
fscore_topic_rf,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', RandomForestClassifier(n_estimators=rf_est_features), Model.FEATURES_CODE)
pd.Series(fscore_topic_rf).to_pickle('results/f1scores_twitterLTO_features_rf.pkl')
print("------------------------------------------------------")

f = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{f1score}', fontsize=11)
df = pd.DataFrame({'NB': fscore_topic_nb, 'KNN': fscore_topic_knn, 'SVM': fscore_topic_svm, 'RF': fscore_topic_rf})
boxplot = df.boxplot(column=['NB', 'KNN', 'SVM', 'RF'])
f.savefig("results/f1score_twitter_features.svg", dpi=100, bbox_inches='tight')
f.clear()

# WORD2VEC
print("WORD2VEC + GAUSSIANNB")
t.custom_gridsearch(dataset=dataset, platform=platform, parameters={}, language=language, clf="gnb", feature_set=Model.WORD2VEC_CODE)
fscore_topic_nb,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', GaussianNB(), Model.WORD2VEC_CODE)
pd.Series(fscore_topic_nb).to_pickle('results/f1scores_twitterLTO_word2vec_nb.pkl')
print("------------------------------------------------------")

print("WORD2VEC + KNN")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__n_neighbors':[1,3,5,7]}, language=language, clf="knn", feature_set=Model.WORD2VEC_CODE)
knn_k_word2vec = params['clf__n_neighbors']
fscore_topic_knn,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', KNeighborsClassifier(n_neighbors=knn_k_word2vec), Model.WORD2VEC_CODE)
pd.Series(fscore_topic_knn).to_pickle('results/f1scores_twitterLTO_word2vec_knn.pkl')
print("------------------------------------------------------")

print("WORD2VEC + SVC")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__kernel':['poly', 'rbf', 'linear', 'sigmoid']}, language=language, clf="svm", feature_set=Model.WORD2VEC_CODE)
svc_kernel_word2vec = params['clf__kernel']
fscore_topic_svm,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', SVC(kernel=svc_kernel_word2vec), Model.WORD2VEC_CODE)
pd.Series(fscore_topic_svm).to_pickle('results/f1scores_twitterLTO_word2vec_svm.pkl')

print("------------------------------------------------------")

print("WORD2VEC + RF")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__n_estimators':tree_estimators}, language=language, clf="rf", feature_set=Model.WORD2VEC_CODE)
rf_est_word2vec = params['clf__n_estimators']
fscore_topic_rf,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', RandomForestClassifier(n_estimators=rf_est_word2vec), Model.WORD2VEC_CODE)
pd.Series(fscore_topic_rf).to_pickle('results/f1scores_twitterLTO_word2vec_rf.pkl')
print("------------------------------------------------------")

f = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{f1score}', fontsize=11)
df = pd.DataFrame({'NB': fscore_topic_nb, 'KNN': fscore_topic_knn, 'SVM': fscore_topic_svm, 'RF': fscore_topic_rf})
boxplot = df.boxplot(column=['NB', 'KNN', 'SVM', 'RF'])
f.savefig("results/f1score_twitter_word2vec.svg", dpi=100, bbox_inches='tight')
f.clear()


# DCDISTANCE
print("DCDISTANCE + GAUSSIANNB")
t.custom_gridsearch(dataset=dataset, platform=platform, parameters={}, language=language, clf="gnb", feature_set=Model.DCDISTANCE_CODE)
fscore_topic_nb,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', GaussianNB(), Model.DCDISTANCE_CODE)
pd.Series(fscore_topic_nb).to_pickle('results/f1scores_twitterLTO_dcdistance_nb.pkl')
print("------------------------------------------------------")

print("DCDISTANCE + KNN")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__n_neighbors':[1,3,5,7]}, language=language, clf="knn", feature_set=Model.DCDISTANCE_CODE)
knn_k_dcdistance = params['clf__n_neighbors']
fscore_topic_knn,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', KNeighborsClassifier(n_neighbors=knn_k_dcdistance), Model.DCDISTANCE_CODE)
pd.Series(fscore_topic_knn).to_pickle('results/f1scores_twitterLTO_dcdistance_knn.pkl')
print("------------------------------------------------------")

print("DCDISTANCE + SVC")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__kernel':['poly', 'rbf', 'linear', 'sigmoid']}, language=language, clf="svm", feature_set=Model.DCDISTANCE_CODE)
svc_kernel_dcdistance = params['clf__kernel']
fscore_topic_svm,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', SVC(kernel=svc_kernel_dcdistance), Model.DCDISTANCE_CODE)
pd.Series(fscore_topic_svm).to_pickle('results/f1scores_twitterLTO_dcdistance_svm.pkl')

print("------------------------------------------------------")

print("DCDISTANCE + RF")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__n_estimators':tree_estimators}, language=language, clf="rf", feature_set=Model.DCDISTANCE_CODE)
rf_est_dcdistance = params['clf__n_estimators']
fscore_topic_rf,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', RandomForestClassifier(n_estimators=rf_est_dcdistance), Model.DCDISTANCE_CODE)
pd.Series(fscore_topic_rf).to_pickle('results/f1scores_twitterLTO_dcdistance_rf.pkl')
print("------------------------------------------------------")

f = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{f1score}', fontsize=11)
df = pd.DataFrame({'NB': fscore_topic_nb, 'KNN': fscore_topic_knn, 'SVM': fscore_topic_svm, 'RF': fscore_topic_rf})
boxplot = df.boxplot(column=['NB', 'KNN', 'SVM', 'RF'])
f.savefig("results/f1score_twitter_dcdistance.svg", dpi=100, bbox_inches='tight')
f.clear()


# BOW
print("BOW + MultinomialNB")
t.custom_gridsearch(dataset=dataset, platform=platform, parameters={}, language=language, clf="mnb", feature_set=Model.BOW_CODE)
fscore_topic_nb,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', MultinomialNB(), Model.BOW_CODE)
pd.Series(fscore_topic_nb).to_pickle('results/f1scores_twitterLTO_bow_nb.pkl')
print("------------------------------------------------------")

print("BOW + KNN")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__n_neighbors':[1,3,5,7]}, language=language, clf="knn", feature_set=Model.BOW_CODE)
knn_k_bow = params['clf__n_neighbors']
fscore_topic_knn,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', KNeighborsClassifier(n_neighbors=knn_k_bow), Model.BOW_CODE)
pd.Series(fscore_topic_knn).to_pickle('results/f1scores_twitterLTO_bow_knn.pkl')
print("------------------------------------------------------")

print("BOW + SVC")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__kernel':['poly', 'rbf', 'linear', 'sigmoid']}, language=language, clf="svm", feature_set=Model.BOW_CODE)
svc_kernel_bow = params['clf__kernel']
fscore_topic_svm,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', SVC(kernel=svc_kernel_bow), Model.BOW_CODE)
pd.Series(fscore_topic_svm).to_pickle('results/f1scores_twitterLTO_bow_svm.pkl')
print("------------------------------------------------------")

print("BOW + RF")
_, params = t.custom_gridsearch(dataset=dataset, platform=platform, parameters={'clf__n_estimators':tree_estimators}, language=language, clf="rf", feature_set=Model.BOW_CODE)
rf_est_bow = params['clf__n_estimators']
fscore_topic_rf,_,_,_,_ = t.classify_lto('Twitter', 'tweets_br', 'pt', RandomForestClassifier(n_estimators=rf_est_bow), Model.BOW_CODE)
pd.Series(fscore_topic_rf).to_pickle('results/f1scores_twitterLTO_bow_rf.pkl')
print("------------------------------------------------------")

f = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{f1score}', fontsize=11)
df = pd.DataFrame({'NB': fscore_topic_nb, 'KNN': fscore_topic_knn, 'SVM': fscore_topic_svm, 'RF': fscore_topic_rf})
boxplot = df.boxplot(column=['NB', 'KNN', 'SVM', 'RF'])
f.savefig("results/f1score_twitter_bow.svg", dpi=100, bbox_inches='tight')
f.clear()

##########################################################################################################
print("##############################################################")
print("##############################################################")
print("##############################################################")
print()

# FEATURES
print("FEATURES + GAUSSIANNB")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=GaussianNB(), feature_set=Model.FEATURES_CODE, kfold=5)
print("------------------------------------------------------")

print("FEATURES + KNN")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=KNeighborsClassifier(n_jobs=2, n_neighbors=knn_k_features), feature_set=Model.FEATURES_CODE, kfold=5)
print("------------------------------------------------------")

print("FEATURES + SVC")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=SVC(random_state=42, gamma='auto', kernel=svc_kernel_features), feature_set=Model.FEATURES_CODE, kfold=5)
print("------------------------------------------------------")

print("FEATURES + RF")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=RandomForestClassifier(n_jobs=2, random_state=42, n_estimators=rf_est_features), feature_set=Model.FEATURES_CODE, kfold=5)
print("------------------------------------------------------")


# WORD2VEC
print("WORD2VEC + GAUSSIANNB")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=GaussianNB(), feature_set=Model.WORD2VEC_CODE, kfold=5)
print("------------------------------------------------------")

print("WORD2VEC + KNN")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=KNeighborsClassifier(n_jobs=2, n_neighbors=knn_k_word2vec), feature_set=Model.WORD2VEC_CODE, kfold=5)
print("------------------------------------------------------")

print("WORD2VEC + SVC")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=SVC(random_state=42, gamma='auto', kernel=svc_kernel_word2vec), feature_set=Model.WORD2VEC_CODE, kfold=5)
print("------------------------------------------------------")

print("WORD2VEC + RF")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=RandomForestClassifier(n_jobs=2, n_estimators=rf_est_word2vec, random_state=42), feature_set=Model.WORD2VEC_CODE, kfold=5)
print("------------------------------------------------------")


# DCDISTANCE
print("DCDISTANCE + GAUSSIANNB")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=GaussianNB(), feature_set=Model.DCDISTANCE_CODE, kfold=5)
print("------------------------------------------------------")

print("DCDISTANCE + KNN")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=KNeighborsClassifier(n_jobs=2, n_neighbors=knn_k_dcdistance), feature_set=Model.DCDISTANCE_CODE, kfold=5)
print("------------------------------------------------------")

print("DCDISTANCE + SVC")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=SVC(random_state=42, gamma='auto', kernel=svc_kernel_dcdistance), feature_set=Model.DCDISTANCE_CODE, kfold=5)
print("------------------------------------------------------")

print("DCDISTANCE + RF")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=RandomForestClassifier(n_jobs=2, n_estimators=rf_est_dcdistance, random_state=42), feature_set=Model.DCDISTANCE_CODE, kfold=5)
print("------------------------------------------------------")


# BOW
print("BOW + MultinomialNB")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=MultinomialNB(), feature_set=Model.BOW_CODE, kfold=5)
print("------------------------------------------------------")

print("BOW + KNN")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=KNeighborsClassifier(n_jobs=2, n_neighbors=knn_k_bow), feature_set=Model.BOW_CODE, kfold=5)
print("------------------------------------------------------")

print("BOW + SVC")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=SVC(random_state=42, gamma='auto', kernel=svc_kernel_bow), feature_set=Model.BOW_CODE, kfold=5)
print("------------------------------------------------------")

print("BOW + RF")
_, params = t.classify(dataset=dataset, platform=platform, language=language, clf=RandomForestClassifier(n_jobs=2, random_state=42, n_estimators=rf_est_bow), feature_set=Model.BOW_CODE, kfold=5)
print("------------------------------------------------------")

