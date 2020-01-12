from preprocessing.preprocessor_twitter import PreProcessorTwitter
from preprocessing.preprocessor_texts import PreProcessorTexts
from glob import glob
import numpy as np
import pandas as pd

DTYPES_TEXT = {'uppercase': np.float64, 'exclamation': np.float64, "has_exclamation": np.float64, 'question': np.float64, 'adj': np.float64, 'adv': np.float64, 'noun': np.float64, 'spell_errors': np.float64, 'lexical_size': np.float64, 'Text': str, 'polarity': np.float64, 'number_sentences': np.float64, 'len_text': np.float64, 'label': np.float64, 'words_per_sentence': np.float64, 'swear_words': np.float64}
FEATURES_TEXT = ["uppercase", "exclamation", "has_exclamation", "question", "adj", "adv", "noun", "spell_errors", "lexical_size", "polarity", "number_sentences", "len_text", "words_per_sentence", "word2vec", "label"]


def setup():
    print("Setting up fakenewsdata1_randomPolitics")
    p = PreProcessorTexts(language='en', embeddings_path='embeddings/en/model.bin', platform='Websites', dataset="fakenewsdata1_randomPolitics")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="fakenewsdata1_randomPolitics", class_label="False")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="fakenewsdata1_randomPolitics", class_label="True")
    print("--------------------------")
    print()

    print("Setting up Bhattacharjee")
    p = PreProcessorTexts(language='en', embeddings_path='embeddings/en/model.bin', platform='Websites', dataset="Bhattacharjee")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="Bhattacharjee", class_label="False")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="Bhattacharjee", class_label="True")
    print("--------------------------")
    print()
    
    print("Setting up btv-lifestyle")
    p = PreProcessorTexts(language='bg', embeddings_path='embeddings/bg/model.txt', platform='Websites', dataset="btv-lifestyle")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="btv-lifestyle", class_label="False")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="btv-lifestyle", class_label="True")
    print("--------------------------")
    print()

    print("Setting up tweets_br")
    p = PreProcessorTwitter(language='pt', embeddings_path='embeddings/pt/model.txt',  dataset="tweets_br")
    p.convert_rawdataset_to_dataset(platform="Twitter", dataset="tweets_br", class_label="False")
    p.convert_rawdataset_to_dataset(platform="Twitter", dataset="tweets_br", class_label="True")
    print("--------------------------")
    print()

    print("Setting up WhatsApp")
    p = PreProcessorTexts(language='pt', embeddings_path='embeddings/pt/model.txt', platform='WhatsApp', dataset="whats_br")
    p.convert_rawdataset_to_dataset(platform="WhatsApp", dataset="whats_br", class_label="False")
    p.convert_rawdataset_to_dataset(platform="WhatsApp", dataset="whats_br", class_label="True")
    print("--------------------------")
    print()
    
    print("Setting up FakeBrCorpus")
    p = PreProcessorTexts(language='pt', embeddings_path='embeddings/pt/model.txt', platform='Websites', dataset="FakeBrCorpus")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="FakeBrCorpus", class_label="False")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="FakeBrCorpus", class_label="True")
    print("--------------------------")
    print()


if __name__ == "__main__":
    setup()
