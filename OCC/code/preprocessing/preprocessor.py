import re
import os
from glob import glob
from abc import ABC, abstractmethod, abstractproperty

import nltk
from nltk.tokenize import RegexpTokenizer
import polyglot
from polyglot.text import Word, Text
from gensim import models
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import hunspell
import stop_words

from nltk.stem import PorterStemmer, RSLPStemmer
from .bulgarianstemmer.bulgarian_stemmer import BulgarianStemmer

np.random.seed(42)

DTYPES_TEXT = {'uppercase': np.float64, 'exclamation': np.float64, "has_exclamation": np.float64, 'question': np.float64, 'adj': np.float64, 'adv': np.float64, 'noun': np.float64, 'spell_errors': np.float64, 'lexical_size': np.float64, 'Text': str, 'polarity': np.float64, 'number_sentences': np.float64, 'len_text': np.float64, 'label': np.float64, 'words_per_sentence': np.float64, 'word2vec': np.float64}
FEATURES_TEXT = ["uppercase", "exclamation", "has_exclamation", "question", "adj", "adv", "noun", "spell_errors", "lexical_size", "polarity", "number_sentences", "len_text", "words_per_sentence", "word2vec", "label"]


class PreProcessor():

    """
    Convert raw to structured data.
    """

    def __init__(self, language: str, platform_folder: str, embeddings_path: str):
        """
        :param platform_folder: str, like datasets/Websites/FakeBrCorpus/
        :param embeddings_path: str, like embeddings/pt/model.txt
        """
        self.PLATFORM_FOLDER = platform_folder
        binary = True if embeddings_path.split('.')[-1] == 'bin' else False
        self.embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=binary, unicode_errors='ignore')
        self.LANGUAGE = language
        self.PT_BR_dic = 'Dictionaries/pt_BR/pt_BR.dic'
        self.PT_BR_aff = 'Dictionaries/pt_BR/pt_BR.aff'
        self.EN_US_dic = 'Dictionaries/en_US/en_US.dic'
        self.EN_US_aff = 'Dictionaries/en_US/en_US.aff'
        self.BG_BG_dic = 'Dictionaries/bg_BG/bg.dic'
        self.BG_BG_aff = 'Dictionaries/bg_BG/bg.aff'
        if language == 'pt':
            self.spell_checker = hunspell.HunSpell(self.PT_BR_dic, self.PT_BR_aff)
        elif language == 'en':
            self.spell_checker = hunspell.HunSpell(self.EN_US_dic, self.EN_US_aff)
        else:
            self.spell_checker = hunspell.HunSpell(self.BG_BG_dic, self.BG_BG_aff)

    def _tokenise(self, text: str) -> list:
        """
        Breaks a text into a list of words.
        Example:
            tokenise("Come here. Again, sir.") -> ['Come', 'here', 'Again', 'sir']
        """
        tokenizer = RegexpTokenizer(r'\w+')
        return tokenizer.tokenize(text)
    
    def _remove_stop_words(self, words: list) -> list:
        """
        Given a list of words, return the same list, but
        without words considered stopwords for the given 
        language.
        """
        return list(filter(lambda x: x not in stop_words.get_stop_words(self.LANGUAGE), words))

    def _stemmer(self, words: list) -> list:
        """
        Given a list of words, returns the list 
        with words in theis stemmed forms.
        """
        if self.LANGUAGE == 'bg':
            stemmer = BulgarianStemmer('preprocessing/bulgarianstemmer/stem_rules_context_2.txt')
            return list(map(stemmer, words))
        if self.LANGUAGE == 'en':
            stemmer = PorterStemmer()
            return list(map(stemmer.stem, words))
        else:  # Portuguese
            stemmer = RSLPStemmer()
            return list(map(stemmer.stem, words))

    def preprocess_text(self, text) -> str:
        lowered = text.lower()
        no_numbers = re.sub('[0-9]+', '', lowered)
        tokenised = self._tokenise(no_numbers)
        stopwords_removed = self._remove_stop_words(tokenised)
        stemmed = self._stemmer(stopwords_removed)
        return " ".join(stemmed)

    def _text_2_list_of_list_of_strings(self, text: str) -> list:
        """
        Break sentences into a list of lists of
        strings.

        Example:

            _text_2_list_of_list_of_strings('First sentence. Now the second one.')

            Returns: [['First','sentence'], ['Now','the','second','one']]

        :param text: any string.
        :return: a list  of lists. Each sublist
        contains words (str format).
        """
        sentences = nltk.sent_tokenize(text)
        strings = []
        tokenizer = RegexpTokenizer(r'\w+')
        for s in sentences:
            strings.append(tokenizer.tokenize(s))
        return strings

    def get_words_per_sentence(self, text: str) -> float:
        """Given a text, returns how many words, in average,
        sentences have.

        :param text: a string.
        :return: a float, with the average size of the sentences.
        """
        return np.mean(list(map(len, self._text_2_list_of_list_of_strings(text))))

    def polarity_text(self, text: str) -> float:
        t = Text(text=text, hint_language_code=self.LANGUAGE)
        try:
            return t.polarity
        except:
            return 0

    def get_proportion_spell_error(self, text: str) -> float:
        """Returns the proportion of words wrongly spelled
        in the given text.

        :param text: a string.
        :return: [0..1].
        """
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
        total = 1
        errors = 0
        for word in words:
            try:
                if self.spell_checker.spell(word) is False:
                    errors += 1
                total += 1
            except UnicodeEncodeError as e:
                print(str(e))
        return errors / total

    def get_lexical_size(self, text: str) -> int:
        """Returns how many unique words there
        are in the given text.
        """
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
        return len(set(words))

    def get_proportion_tag(self, text: str, tag='ADJ') -> float:
        """
        Returns the proportion of tokens in a given text
        that are of type <tag>.

        For example, if <tag> is ADJ, and 3 of 10 tokens are ADJ,
        it will return 0.3
        :param text: any string.
        :param tag: string, such as ADJ, ADV, NOUN...
        """
        t = Text(text=text, hint_language_code=self.LANGUAGE)
        pos_tags = t.pos_tags
        return len(list(filter(lambda x: x[1] == tag, pos_tags))) / len(pos_tags)

    def custom_doc2vec(self, text: str, printing=False) -> np.ndarray:
        """
        Using a pre-trained word2vec model,
        it gets all possible word vectors
        from a text. Returns the sum of these
        vectors.
        """
        words = set()
        for sentence in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sentence):
                words.add(word.lower())
        word_vectors = []
        number_words = len(words)
        missing = 0
        for word in words:
            try:
                word_vector = self.embeddings.get_vector(word)
                word_vectors.append(word_vector)
            except KeyError:
                missing += 1
        if printing:
            print("Missing {0}/{1} words".format(missing, number_words))
        return np.array(word_vectors).sum(axis=0)

    def get_word2vec_mean(self, text) -> float:
        return self.custom_doc2vec(text, printing=True).mean()

    @abstractmethod
    def _load_file(self, filepath: str) -> dict:
        pass

    @abstractmethod
    def _convert_json_to_obj(self, json: dict):
        pass

    def convert_rawdataset_to_dataset(self, platform: str, dataset: str, class_label: str):
        """
        This method sould parse all .txt content in Raw/class_label/ folders and store
        the content into Structured/class_label/ folders in .csv format.

        :param platform: either 'Twitter', 'Website' or 'WhatsApp'
        :dataset: 'tweets_br', 'br'
        :param class_label: either 'True' of 'False'
        """
        dataset_dataframe = pd.DataFrame()
        for filepath in sorted(glob('datasets/{0}/{1}/Raw/{2}/*/*'.format(platform, dataset, class_label))):
            # filepath = datasets/WhatsApp/br/Raw/False/Fake topic/1020479528047726593.txt
            obj_dict = self._load_file(filepath=filepath)
            obj = self._convert_json_to_obj(obj_dict)
            if obj is not None:
                df = self.convert_obj_to_dataframe(label=class_label, obj=obj, platform=platform)
                dataset_dataframe = dataset_dataframe.append(df)
            else:
                print("convert_rawdataset_to_dataset: failed to convert {0}".format(filepath))
        dataset_dataframe.to_csv('datasets/{0}/{1}/Structured/features_{2}.csv'.format(platform, dataset, class_label), columns=FEATURES_TEXT, header=False, index=False)
        doc2vecs = pd.DataFrame(np.array(list(map(self.custom_doc2vec, dataset_dataframe.Text))))
        doc2vecs.insert(doc2vecs.shape[-1], "label", dataset_dataframe.label.values)
        doc2vecs.to_csv('datasets/{0}/{1}/Structured/word2vec_{2}.csv'.format(platform, dataset, class_label), header=False, index=False)
        txtdataset = dataset_dataframe.Text.to_frame()
        txtdataset.insert(txtdataset.shape[-1], "label", dataset_dataframe.label.values)
        txtdataset.to_csv('datasets/{0}/{1}/Structured/text_{2}.csv'.format(platform, dataset, class_label), header=False, index=False)

    @abstractmethod
    def convert_obj_to_dataframe(self, label: str, obj, platform: str) -> pd.DataFrame():
        pass
