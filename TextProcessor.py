import re
from collections import defaultdict

import numpy as np
from nltk import PorterStemmer


class TextProcessor:
    def __init__(self, text):
        self.text = text
        self.tokens = None

    def tokenize(self):
        self.tokens = self.text.strip().split()
        return self

    def lowercase_folding(self):
        if self.tokens:
            self.tokens = [token.lower() for token in self.tokens]
        return self

    def remove_non_alphanumeric(self):
        if self.tokens:
            self.tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in self.tokens]
        return self

    def porter_stemming(self):
        if self.tokens:
            porter = PorterStemmer()
            self.tokens = [
                porter.stem(token) for token in self.tokens
            ]
        return self

    def word_count(self):
        count_dict = defaultdict(int)
        for token in self.tokens:
            count_dict[token] += 1
        return dict(count_dict)

    def save_to_file(self, file_name='output.txt'):
        if self.tokens:
            with open('Dataset/TextProcessing/' + file_name, 'w', encoding='utf8') as file:
                file.write(' '.join(self.tokens))
        return self

    def process(self):
        return self\
            .tokenize()\
            .lowercase_folding()\
            .remove_non_alphanumeric()\
            .porter_stemming()\
            .save_to_file()
