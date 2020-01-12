# python boxplots
# Usado para a dissertacao. Gera boxplots do LTO com labels maiores.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = "tweets_br"
platform = 'Twitter'
language = "pt"
tree_estimators = np.arange(1, 1001, 50)




f = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{f1score}', fontsize=22)
fscore_topic_nb = pd.read_pickle("results/f1scores_twitterLTO_features_nb.pkl")
fscore_topic_knn = pd.read_pickle("results/f1scores_twitterLTO_features_knn.pkl")
fscore_topic_svm = pd.read_pickle("results/f1scores_twitterLTO_features_svm.pkl")
fscore_topic_rf = pd.read_pickle("results/f1scores_twitterLTO_features_rf.pkl")
df = pd.DataFrame({'NB': fscore_topic_nb, 'KNN': fscore_topic_knn, 'SVM': fscore_topic_svm, 'RF': fscore_topic_rf})
boxplot = df.boxplot(column=['NB', 'KNN', 'SVM', 'RF'], fontsize=22)
f.savefig("results/f1score_twitter_features.pdf", dpi=100, bbox_inches='tight')
f.clear()


f = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{f1score}', fontsize=22)
fscore_topic_nb = pd.read_pickle("results/f1scores_twitterLTO_word2vec_nb.pkl")
fscore_topic_knn = pd.read_pickle("results/f1scores_twitterLTO_word2vec_knn.pkl")
fscore_topic_svm = pd.read_pickle("results/f1scores_twitterLTO_word2vec_svm.pkl")
fscore_topic_rf = pd.read_pickle("results/f1scores_twitterLTO_word2vec_rf.pkl")
df = pd.DataFrame({'NB': fscore_topic_nb, 'KNN': fscore_topic_knn, 'SVM': fscore_topic_svm, 'RF': fscore_topic_rf})
boxplot = df.boxplot(column=['NB', 'KNN', 'SVM', 'RF'], fontsize=22)
f.savefig("results/f1score_twitter_word2vec.pdf", dpi=100, bbox_inches='tight')
f.clear()


f = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{f1score}', fontsize=22)
fscore_topic_nb = pd.read_pickle("results/f1scores_twitterLTO_dcdistance_nb.pkl")
fscore_topic_knn = pd.read_pickle("results/f1scores_twitterLTO_dcdistance_knn.pkl")
fscore_topic_svm = pd.read_pickle("results/f1scores_twitterLTO_dcdistance_svm.pkl")
fscore_topic_rf = pd.read_pickle("results/f1scores_twitterLTO_dcdistance_rf.pkl")
df = pd.DataFrame({'NB': fscore_topic_nb, 'KNN': fscore_topic_knn, 'SVM': fscore_topic_svm, 'RF': fscore_topic_rf})
boxplot = df.boxplot(column=['NB', 'KNN', 'SVM', 'RF'], fontsize=22)
f.savefig("results/f1score_twitter_dcdistance.pdf", dpi=100, bbox_inches='tight')
f.clear()



f = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{f1score}', fontsize=22)
fscore_topic_nb = pd.read_pickle("results/f1scores_twitterLTO_bow_nb.pkl")
fscore_topic_knn = pd.read_pickle("results/f1scores_twitterLTO_bow_knn.pkl")
fscore_topic_svm = pd.read_pickle("results/f1scores_twitterLTO_bow_svm.pkl")
fscore_topic_rf = pd.read_pickle("results/f1scores_twitterLTO_bow_rf.pkl")
df = pd.DataFrame({'NB': fscore_topic_nb, 'KNN': fscore_topic_knn, 'SVM': fscore_topic_svm, 'RF': fscore_topic_rf})
boxplot = df.boxplot(column=['NB', 'KNN', 'SVM', 'RF'], fontsize=22)
f.savefig("results/f1score_twitter_bow.pdf", dpi=100, bbox_inches='tight')
f.clear()

