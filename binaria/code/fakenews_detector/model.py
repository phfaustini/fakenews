import warnings
from glob import glob
import random
from sys import maxsize

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from scipy.spatial.distance import cosine
from stop_words import get_stop_words

from .dcdistance import DCDistance
from .preprocessor import PreProcessor

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)


class Model():

    N_JOBS = 2
    RANDOM_STATE = 42
    FEATURES_CODE = 1
    BOW_CODE = 2
    DCDISTANCE_CODE = 3
    DCDISTANCE_BOW_FEATURES_CODE = 4
    WORD2VEC_CODE = 5
    DTYPES_TEXT = {'uppercase': np.float64, 'exclamation': np.float64, "has_exclamation": np.float64, 'question': np.float64, 'adj': np.float64, 'adv': np.float64, 'noun': np.float64, 'spell_errors': np.float64, 'lexical_size': np.float64, 'Text': str, 'polarity': np.float64, 'number_sentences': np.float64, 'len_text': np.float64, 'word2vec': np.float64, 'label': np.float64, 'words_per_sentence': np.float64}
    FEATURES_TEXT = ["uppercase", "exclamation", "has_exclamation", "question", "adj", "adv", "noun", "spell_errors", "lexical_size", "polarity", "number_sentences", "len_text", "words_per_sentence", "word2vec"]

    def __init__(self, language: str, platform_folder: str, embeddings_path: str):
        self.data = None
        self.preprocessor = PreProcessor(language=language, platform_folder=platform_folder, embeddings_path=embeddings_path)

    def _get_word2vec_dataset(self, documents: np.ndarray) -> np.ndarray:
        """
        :param documents: a list of strings.
        :return: array like (n_documents, n_features)
        """
        doc2vecs = list(map(self.preprocessor.custom_doc2vec, documents))
        return np.array(doc2vecs)

    def load_data(self, folders: list, dtypes=DTYPES_TEXT) -> pd.DataFrame():
        """
        Returns a pd.DataFrame with objects of all .csv files
        inside folders listed in folders parameter.

        :param folder: a list of strings list of folders
            e.g.: 'datasets/WhatsApp/br/Structured/False/'
        :returns: a pd.DataFrame with objects' data.
        """
        df = pd.DataFrame()
        c = 0
        for folder in folders:
            files = sorted(glob("{0}/*".format(folder)))
            for f in files:
                try:
                    pd_obj = pd.read_csv("{0}".format(f), index_col=False, dtype=dtypes, header=0)
                    pd_obj['topic'] = [c]
                    df = df.append(pd_obj)
                except:
                    # self.logger.warning("load_data: bad .csv {0}/{1}".format(root, f), low_memory=False)
                    pass
            c += 1
        df.index = range(df.shape[0])
        self.data = df
        return self.data

    def classify(self,
                 dataset='FakeBrCorpus',
                 platform='Websites',
                 language='pt',
                 clf=None,
                 parameters={},
                 feature_set=FEATURES_CODE,
                 features=FEATURES_TEXT,
                 dtypes=DTYPES_TEXT,
                 kfold=5) -> tuple:
        """Performs classification.
        :return: a tuple. The first element is the fitted model returned by
        GridSearchCV, and the second is a dict with the best parameters found.
        """
        if self.data is None:
            folders = sorted(glob('datasets/{0}/{1}/Structured/*/*/'.format(platform, dataset)))
            self.load_data(folders=folders, dtypes=dtypes)
        y = self.data.label.values
        scoring = {'accuracy': make_scorer(accuracy_score), 'fscore_false': make_scorer(f1_score, average='binary', pos_label=0)}
        if feature_set == Model.FEATURES_CODE:
            X = self.data.loc[:, features]
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])  # https://stackoverflow.com/questions/48726695/error-when-using-scikit-learn-to-use-pipelines, https://stackoverflow.com/questions/51459406/apply-standardscaler-in-pipeline-in-scikit-learn-sklearn/51465479
            model = GridSearchCV(estimator=pipe, scoring=scoring, refit='accuracy', return_train_score=False, param_grid=parameters, n_jobs=Model.N_JOBS, cv=StratifiedKFold(n_splits=kfold, random_state=Model.RANDOM_STATE))
            model.fit(X, y)
            self._print_results_cv(model)
            return model, model.best_params_
        elif feature_set == Model.BOW_CODE:
            corpus = self.data.Text
            pipe = Pipeline([('bow', TfidfVectorizer(use_idf=True)), ('clf', clf)])
            model = GridSearchCV(estimator=pipe, scoring=scoring, refit='accuracy', return_train_score=False, param_grid=parameters, n_jobs=Model.N_JOBS, cv=StratifiedKFold(n_splits=kfold, random_state=Model.RANDOM_STATE))
            model.fit(corpus, y)
            self._print_results_cv(model)
            return model, model.best_params_
        elif feature_set == Model.DCDISTANCE_CODE:
            corpus = self.data.Text
            pipe = Pipeline([('bow', TfidfVectorizer(use_idf=True)), ('dcdistance', DCDistance(distance=cosine)), ('clf', clf)])
            model = GridSearchCV(estimator=pipe, scoring=scoring, refit='accuracy', return_train_score=False, param_grid=parameters, n_jobs=Model.N_JOBS, cv=StratifiedKFold(n_splits=kfold, random_state=Model.RANDOM_STATE))
            model.fit(corpus, y)
            self._print_results_cv(model)
            return model, model.best_params_
        elif feature_set == Model.WORD2VEC_CODE:
            X_word2vec = self._get_word2vec_dataset(self.data.Text)
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
            model = GridSearchCV(estimator=pipe, scoring=scoring, refit='accuracy', return_train_score=False, param_grid=parameters, n_jobs=Model.N_JOBS, cv=StratifiedKFold(n_splits=kfold, random_state=Model.RANDOM_STATE))
            model.fit(X_word2vec, y)
            self._print_results_cv(model)
            return model, model.best_params_

    def _print_results_cv(self, model) -> None:
        fscore_mean = model.cv_results_['mean_test_fscore_false'][model.best_index_]*100
        fscore_std = model.cv_results_['std_test_fscore_false'][model.best_index_]*100
        acc_mean = model.cv_results_['mean_test_accuracy'][model.best_index_]*100
        acc_std = model.cv_results_['std_test_accuracy'][model.best_index_]*100
        print("Fscore (fake) = {0:.2f}% \u00B1 {1:.2f}%".format(fscore_mean, fscore_std))
        print("Accuracy =      {0:.2f}% \u00B1 {1:.2f}%".format(acc_mean, acc_std))
        print("Best params:    {0}".format(model.best_params_))

    def feature_importance(self, n_estimators: int, platform='Twitter', dataset='tweets_br') -> pd.DataFrame():
        if self.data is None:
            folders = sorted(glob('datasets/{0}/{1}/Structured/*/*/'.format(platform, dataset)))
            self.load_data(folders=folders, dtypes=Model.DTYPES_TEXT)
        y = self.data.label.values
        X = self.data.loc[:, Model.FEATURES_TEXT]
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X, y)
        feature_importances = pd.DataFrame(clf.feature_importances_, index=Model.FEATURES_TEXT, columns=['importance']).sort_values('importance', ascending=False)
        feature_importances.plot(kind='bar', fontsize=18, legend=False)
        plt.tight_layout()
        plt.savefig('results/{0}.pdf'.format(dataset))
        return feature_importances

########################################################################################################################

    def classify_lto(self,
                     platform='Twitter',
                     dataset='tweets_br',
                     language='pt',
                     clf=None,
                     feature_set=FEATURES_CODE,
                     features=FEATURES_TEXT,
                     dtypes=DTYPES_TEXT) -> tuple:
        """
        Leave-one-out strategy: one folder os news in each
        turn is classified.
        :returns: a tuple (y_validation, y_pred), where
        y_validation, y_pred are np.arrays
        """
        fscore_topic = np.array([])
        accuracy_topic = np.array([])
        if self.data is None:
            folders = sorted(glob('datasets/{0}/{1}/Structured/*/*/'.format(platform, dataset)))
            self.load_data(folders=folders, dtypes=dtypes)
        false_data = self.data.loc[self.data['label'] == 0]
        true_data = self.data.loc[self.data['label'] == 1]
        false_topics = set(false_data.topic.values)
        true_topics = set(true_data.topic.values)
        len_false = len(false_topics)
        len_true = len(true_topics)
        topics = set(self.data.topic.values)
        for topic in topics:
            validation_data = self.data.loc[self.data['topic'] == topic]
            validation_other_i = self._get_other_validation_fold(topic, validation_data.label.iloc[0])
            validation_data_otherclass = self.data.loc[self.data['topic'] == validation_other_i]
            validation_data = validation_data.append(validation_data_otherclass, ignore_index=True)
            training_data = self.data.loc[~self.data['topic'].isin([topic, validation_other_i])]
            if feature_set == Model.FEATURES_CODE:
                results = self._classify_with_features(features=features, training_data=training_data, validation_data=validation_data, clf=clf)
            elif feature_set == Model.BOW_CODE:
                results = self._classify_with_bow(language=language, clf=clf, training_data=training_data, validation_data=validation_data)
            elif feature_set == Model.DCDISTANCE_CODE:
                results = self._classify_with_dcdistance(features=['dcdistance_fake', 'dcdistance_true'], training_data=training_data, validation_data=validation_data, clf=clf)
            elif feature_set == Model.WORD2VEC_CODE:
                results = self._classify_with_word2vec(clf=clf, training_data=training_data, validation_data=validation_data)
            fscore_topic = np.append(fscore_topic, f1_score(y_true=results[0], y_pred=results[1], average='binary', pos_label=0))
            accuracy_topic = np.append(accuracy_topic, accuracy_score(y_true=results[0], y_pred=results[1]))
            #print("Topic A: ", topic)
            #print("Topic B: ", validation_other_i)
            #print("Ytrue: ", results[0])
            #print()
            #print("YPred: ", results[1])
            #print("----------------------------")
            #print()
        accuracy_mean, accuracy_std, fscore_mean, fscore_std = self._compute_results(fscore_topic=fscore_topic, accuracy_topic=accuracy_topic)
        return (fscore_topic, accuracy_mean, accuracy_std, fscore_mean, fscore_std)

    def _get_other_validation_fold(self, topic: int, label: int) -> int:
        validation_data = self.data.loc[self.data['topic'] == topic]
        validation_size = validation_data.shape[0]
        others = self.data.loc[self.data['label'] != label]
        topics = set(others.topic.values)
        topics_size = {}
        for topic in topics:
            s = others.loc[self.data['topic'] == topic]
            topics_size[topic] = s.shape[0]
        delta = maxsize
        other_topic = None
        for i, v in topics_size.items():
            if abs(validation_size - v) <= delta:
                other_topic = i
                delta = abs(validation_size - v)
        return other_topic

    def _classify_with_word2vec(self, training_data: pd.DataFrame, validation_data: pd.DataFrame, clf=None) -> tuple:
        corpus = training_data.Text
        y_training = training_data.label
        X_training = self._get_word2vec_dataset(corpus)
        clf.fit(X_training, y_training)

        X_validation = validation_data.Text
        X_validation_transformed = self._get_word2vec_dataset(X_validation)
        y_validation = validation_data.label

        y_pred = clf.predict(X_validation_transformed)
        return (y_validation.values, y_pred)

    def _classify_with_bow(self, training_data: pd.DataFrame, validation_data: pd.DataFrame, clf=None, language='pt') -> tuple:
        """
        Classify data given a classifier clf.
        :returns: a tuple (y_validation, y_pred), where
        y_validation, y_pred are np.arrays
        """
        corpus = training_data.Text
        y_training = training_data.label
        tfidf_vectorizer = TfidfVectorizer(use_idf=True)
        X_training = tfidf_vectorizer.fit_transform(corpus)
        clf.fit(X_training, y_training)

        X_validation = validation_data.Text
        X_validation_transformed = tfidf_vectorizer.transform(X_validation)
        y_validation = validation_data.label

        y_pred = clf.predict(X_validation_transformed)
        return (y_validation.values, y_pred)

    def _classify_with_dcdistance(self, features: list, training_data: pd.DataFrame, validation_data: pd.DataFrame, clf=None) -> tuple:
        """
        Classify data given a classifier clf.
        :returns: a tuple (y_validation, y_pred), where
        y_validation, y_pred are np.arrays
        """
        tfidf_vectorizer = TfidfVectorizer(use_idf=True)
        train_bow = tfidf_vectorizer.fit_transform(training_data.Text)
        test_bow = tfidf_vectorizer.transform(validation_data.Text)

        dcdistance = DCDistance(distance=cosine)
        train_bow_transformed = dcdistance.fit_transform(train_bow, training_data.label.values)
        test_bow_transformed = dcdistance.transform(test_bow)

        clf.fit(train_bow_transformed, training_data.label)
        y_pred = clf.predict(test_bow_transformed)
        return (validation_data.label.values, y_pred)

    def _classify_with_features(self, features: list, training_data: pd.DataFrame, validation_data: pd.DataFrame, clf=None) -> tuple:
        """
        Classify data given a classifier clf.
        :returns: a tuple (y_validation, y_pred), where
        y_validation, y_pred are np.arrays
        """
        X_training = training_data.loc[:, features]
        y_training = training_data.label
        scaler = StandardScaler()
        X_training_transformed = scaler.fit_transform(X_training)

        X_validation = validation_data.loc[:, features]
        X_validation_transformed = scaler.transform(X_validation)
        y_validation = validation_data.label

        clf.fit(X_training_transformed, y_training)
        y_pred = clf.predict(X_validation_transformed)
        return (y_validation.values, y_pred)

    def _compute_results(self, fscore_topic: np.array, accuracy_topic: np.array) -> tuple:
        """
        :return: accuracy score
        """
        accuracy_mean = np.mean(accuracy_topic)*100
        fscore_mean = np.mean(fscore_topic)*100
        fscore_std = np.std(fscore_topic)*100
        accuracy_std = np.std(accuracy_topic)*100
        # print("Fscore (fake) = {0:.2f}% \u00B1 {1:.2f}%".format(fscore_mean, fscore_std))
        # print("Accuracy = {0:.2f}% \u00B1 {1:.2f}%".format(accuracy_mean, accuracy_std))
        return accuracy_mean, accuracy_std, fscore_mean, fscore_std

    def custom_gridsearch(self,
                          platform='Twitter',
                          dataset='tweets_br',
                          language='pt',
                          clf=None,
                          parameters={},
                          feature_set=FEATURES_CODE,
                          features=FEATURES_TEXT,
                          dtypes=DTYPES_TEXT) -> tuple:
        if parameters == {}: parameters = {'ignore': [None]}
        keys = list(parameters.keys())
        keys.append('accuracy_mean')
        keys.append('fscore_mean')
        keys.append('accuracy_std')
        keys.append('fscore_std')
        results = pd.DataFrame(columns=keys)

        for _, values in parameters.items():
            for v in values:
                if clf == 'knn':
                    c = KNeighborsClassifier(n_jobs=Model.N_JOBS, n_neighbors=v)
                elif clf == 'rf':
                    c = RandomForestClassifier(n_jobs=Model.N_JOBS, n_estimators=v, random_state=Model.RANDOM_STATE)
                elif clf == 'svm':
                    c = SVC(kernel=v, gamma='auto', random_state=Model.RANDOM_STATE)
                elif clf == 'gnb':
                    c = GaussianNB()
                else:
                    c = MultinomialNB()
                _, accuracy_mean, accuracy_std, fscore_mean, fscore_std = self.classify_lto(platform, dataset, language, c, feature_set, features, dtypes)
                row = pd.DataFrame(data=[[v, accuracy_mean, fscore_mean, accuracy_std, fscore_std]], columns=keys)
                results = results.append(row, ignore_index=True)

        best_params_idx = results['accuracy_mean'].astype('float64').idxmax()
        best_params = results.iloc[best_params_idx]

        print(best_params)
        return results, best_params
