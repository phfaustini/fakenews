import os
from glob import glob
from abc import ABC, abstractmethod, abstractproperty

import nltk
from nltk.tokenize import RegexpTokenizer
import polyglot
from polyglot.text import Word, Text
from gensim import models
from gensim.models import KeyedVectors
import nltk
import numpy as np
import pandas as pd
import hunspell

np.random.seed(42)

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
        counter = 1
        for filepath in sorted(glob('datasets/{0}/{1}/Raw/{2}/*/*'.format(platform, dataset, class_label))):
            # filepath = datasets/WhatsApp/br/Raw/False/Fake topic/1020479528047726593.txt
            obj_dict = self._load_file(filepath=filepath)
            obj = self._convert_json_to_obj(obj_dict)
            if obj is not None:
                df = self.convert_obj_to_dataframe(label=class_label, obj=obj, platform=platform)
                queryfolder = "{0}/{1}/".format(class_label, filepath.split("/")[5])
                self.write_to_disk(queryfolder, class_label, counter, df)
                counter += 1
            else:
                print("convert_rawdataset_to_dataset: failed to convert {0}".format(filepath))

    @abstractmethod
    def convert_obj_to_dataframe(self, label: str, obj, platform: str) -> pd.DataFrame():
        pass

    def write_to_disk(self, queryfolder: str, label: str, counter: int, df_obj: pd.DataFrame):
        """Save the object to disk, into folder
        datasets/<Twitter,Websites,WhatsApp>/<Something>/

        Files are saved in format <label>-<counter>.csv
        Values in each line are separated by comma (',').

        :param queryfolder: a string ending with '/' (e.g. 'False/Some fake news/')
        :param label: a string with the class of the object.
        :param counter: a unique counting identifier for naming purposes.
        :param df_obj: a pd.DataFrame with structured objects.
        """

        filename = "{0}-{1}.csv".format(label, counter)
        # print("write_to_disk {0}".format(filename))
        if not os.path.isdir("{0}Structured/".format(self.PLATFORM_FOLDER)):
            os.mkdir("{0}Structured/".format(self.PLATFORM_FOLDER))
        if not os.path.isdir("{0}Structured/{1}".format(self.PLATFORM_FOLDER, label)):
            os.mkdir("{0}Structured/{1}".format(self.PLATFORM_FOLDER, label))
        if not os.path.isdir("{0}Structured/{1}".format(self.PLATFORM_FOLDER, queryfolder)):
            os.mkdir("{0}Structured/{1}".format(self.PLATFORM_FOLDER, queryfolder))
        df_obj.to_csv("{0}Structured/{1}{2}".format(self.PLATFORM_FOLDER, queryfolder, filename), index=False)
