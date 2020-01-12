import os
import sys
from glob import glob
from ast import literal_eval

import nltk
import numpy as np
import pandas as pd

from .preprocessor import PreProcessor
from .news import Tweet, User, Text


class PreProcessorTexts(PreProcessor):

    """
    Convert semi-structured (json) to class objects.
    """

    def __init__(self, language: str, platform: str, dataset: str, embeddings_path: str):
        """
        :param platform: either 'WhatsApp' or 'Websites'
        :param dataset: 'br'
        """
        PreProcessor.__init__(self, language=language, embeddings_path=embeddings_path, platform_folder="datasets/{0}/{1}/".format(platform, dataset))

    def _load_file(self, filepath: str) -> dict:
        wp_dict = {'text': "", 'filepath': ""}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                wp_dict['text'] = f.read()
                wp_dict['filepath'] = filepath
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='cp1252') as f:
                wp_dict['text'] = f.read()
                wp_dict['filepath'] = filepath
        return wp_dict

    def _convert_json_to_obj(self, json: dict) -> Text():
        obj = Text()
        text = json['text']
        uppercase_letters_count = len(list(filter(lambda x: str.isupper(x), text)))
        lowercase_letters_count = len(list(filter(lambda x: str.islower(x), text)))
        exclamation_marks_count = len(list(filter(lambda x: x == "!", text)))
        question_marks_count = len(list(filter(lambda x: x == "?", text)))
        total_letters = uppercase_letters_count + lowercase_letters_count
        obj.uppercase_letters = (uppercase_letters_count / total_letters)
        obj.lowercase_letters = (lowercase_letters_count / total_letters)
        obj.exclamation_marks = (exclamation_marks_count / (total_letters + exclamation_marks_count))
        if obj.exclamation_marks > 0:
            obj.has_exclamation = 1.0
        else:
            obj.has_exclamation = 0.0
        obj.question_marks = (question_marks_count / (total_letters + question_marks_count))
        obj.words_per_sentence = self.get_words_per_sentence(text)
        obj.ADJ = self.get_proportion_tag(text, 'ADJ')
        obj.ADV = self.get_proportion_tag(text, 'ADV')
        obj.NOUN = self.get_proportion_tag(text, 'NOUN')
        obj.spell_errors = self.get_proportion_spell_error(text)
        obj.lexical_size = self.get_lexical_size(text)
        obj.polarity = self.polarity_text(text)
        obj.text = text
        obj.number_sentences = float(len(nltk.sent_tokenize(obj.text)))
        obj.len_text = float(len(text))
        obj.word2vec = self.get_word2vec_mean(text)
        return obj

    def convert_obj_to_dataframe(self, label: str, obj: Text, platform: str) -> pd.DataFrame():
        return pd.DataFrame(
                            [
                              [
                                round(obj.uppercase_letters, 2),
                                round(obj.exclamation_marks, 2),
                                obj.has_exclamation,
                                round(obj.question_marks, 2),
                                round(obj.ADJ, 2),
                                round(obj.ADV, 2),
                                round(obj.NOUN, 2),
                                round(obj.spell_errors, 2),
                                round(obj.lexical_size, 2),
                                obj.polarity,
                                obj.number_sentences,
                                obj.len_text,
                                round(obj.words_per_sentence, 2),
                                obj.word2vec,
                                self.preprocess_text(obj.text),
                                1.0 if label == "False" else -1.0
                              ]
                            ],
                            columns=["uppercase", "exclamation", "has_exclamation", "question", "adj", "adv", "noun", "spell_errors", "lexical_size", "polarity", "number_sentences", "len_text", "words_per_sentence", "word2vec", "Text", "label"]
                           )
