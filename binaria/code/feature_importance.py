"""
See results folder to see which hyperparameters achieved best results.
"""

import numpy as np
import matplotlib.pyplot as plt

from fakenews_detector.model import Model


def get_feature_importance(language: str, embeddings_path: str, platform_folder: str, n_estimators: int, platform: str, dataset: str):
    print("#################")
    print(dataset)
    print()
    t = Model(language=language, embeddings_path=embeddings_path, platform_folder=platform_folder)
    feature_importances = t.feature_importance(n_estimators=n_estimators, platform=platform, dataset=dataset)
    print(feature_importances)
    print()


if __name__ == "__main__":
    language = 'en'
    platform_folder='datasets/Websites/fakenewsdata1_randomPolitics/'
    platform = 'Websites'
    dataset = 'fakenewsdata1_randomPolitics'
    embeddings_path='embeddings/en/model.bin'
    n_estimators = 651
    get_feature_importance(language=language, embeddings_path=embeddings_path, platform_folder=platform_folder, n_estimators=n_estimators, platform=platform, dataset=dataset)

    language = 'en'
    platform_folder='datasets/Websites/Bhattacharjee/'
    platform = 'Websites'
    dataset = 'Bhattacharjee'
    embeddings_path='embeddings/en/model.bin'
    n_estimators = 851
    get_feature_importance(language=language, embeddings_path=embeddings_path, platform_folder=platform_folder, n_estimators=n_estimators, platform=platform, dataset=dataset)

    language = 'pt'
    platform_folder='datasets/Websites/FakeBrCorpus/'
    platform = 'Websites'
    dataset = 'FakeBrCorpus'
    embeddings_path='embeddings/pt/model.txt'
    n_estimators = 601
    get_feature_importance(language=language, embeddings_path=embeddings_path, platform_folder=platform_folder, n_estimators=n_estimators, platform=platform, dataset=dataset)


    language = 'pt'
    platform_folder='datasets/Websites/tweets_br/'
    platform = 'Twitter'
    dataset = 'tweets_br'
    embeddings_path='embeddings/pt/model.txt'
    n_estimators = 201
    get_feature_importance(language=language, embeddings_path=embeddings_path, platform_folder=platform_folder, n_estimators=n_estimators, platform=platform, dataset=dataset)


    language = 'bg'
    platform_folder='datasets/Websites/btv-lifestyle/'
    platform = 'Websites'
    dataset = 'btv-lifestyle'
    embeddings_path='embeddings/bg/model.txt'
    n_estimators = 101
    get_feature_importance(language=language, embeddings_path=embeddings_path, platform_folder=platform_folder, n_estimators=n_estimators, platform=platform, dataset=dataset)
