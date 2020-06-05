import pickle
import os
import string
import pandas as pd
import re
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
import numpy as np
from copy import deepcopy
import random
from nltk.stem import PorterStemmer
from collections import Counter
import joblib

from .models import Dataset, ModelVectorSpace, KNNClasification

FILE_PATH = os.path.dirname(__file__) + '../../data/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOPWORD_PATH = ('Stopword-List.txt')
# Remove Punctuation

printable = set(string.printable)


def remove_punctuation(word):
    return word.translate(word.maketrans('', '', string.punctuation))


# Clean Query Term
def clean_word(word):
    # Case Folding
    ps = PorterStemmer()
    word = word.lower()
    # Filter non-ASCII characters
    word = ''.join(filter(lambda x: x in printable, word))
    #     print(word)
    # Remove Punctuations
    if word != '(' and word != ')':
        word = remove_punctuation(word)
#     print(word)
    if re.match('\d+[A-Za-z]+', word):
        word = re.split('\d+', word)[1]
    if re.match('[A-Za-z]+\d+', word):
        word = re.split('\d+', word)[0]


#     print(word)
    word = ps.stem(word)
    #     print(word)
    return word


class DocToVec(object):
    def __init__(self, DOCUMENTS_PATH, STOP_WORD_PATH):
        self.doc_index = {}
        self.documents_path = DOCUMENTS_PATH
        self.stop_word_path = STOP_WORD_PATH
        self.stop_words = self.load_stop_words()
        self.vocab_index = self.file_extraction_wrapper(extract_vocab=True)
        self.vectors = self.file_extraction_wrapper(extract_vectors=True)
        self.X = self.vectors[0]
        self.y = self.vectors[1]

        data = pd.DataFrame(self.X)
        # # Feature Selection
        # Drop Features with Df < 3
        data.drop([
            col for col, val in data.sum().iteritems() if int(val) <= 3
        ], axis=1, inplace=True)
        data.mul(data.sum().apply(lambda df: np.log10(data.shape[0] / (df + 1))),
                 axis=1)
        self.data = data
        # Tf - Idf Calculation
        self.idf = data.sum().apply(
            lambda df: np.log10(data.shape[0] / (df + 1)))

    def file_extraction_wrapper(self,
                                extract_vocab=False,
                                extract_vectors=False):
        vocab = set()
        docs = {}
        printable = set(string.printable)
        raw_data = []
        if extract_vectors:
            X = []
            y = []
        doc_count = 0
        # Printable characters are
        # 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
        # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c
        ps = PorterStemmer()
        classes = next(os.walk(os.path.join(self.documents_path)))[1]
        for c in classes:
            docs_in_c = next(os.walk(os.path.join(self.documents_path, c)))[2]
            print(c)
            for doc in docs_in_c:
                #             print(f'\t{doc}',end=',')
                if extract_vectors:
                    doc_vector = np.zeros((len(self.vocab_index)))
                with open(os.path.join(self.documents_path, c, doc),
                          'r') as file1:
                    lines = file1.readlines()
                    if extract_vectors:
                        doc_name = os.path.join(self.documents_path, c, doc)
                        self.doc_index[doc_count] = os.path.join(c, doc)
                        doc_count += 1

                    for line_no, line in enumerate(lines):

                        #                         symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
                        #                         for i in symbols:
                        #                             line = line.replace(i, ' ')
                        for word in re.split('[.\s,?!:;-]', line):

                            # Case Folding
                            word = word.lower()

                            # Filter non-ASCII characters
                            word = ''.join(
                                filter(lambda x: x in printable, word))

                            if word in self.stop_words:
                                continue

                            # Remove Punctuations
                            word = remove_punctuation(word)

                            if re.match('\d+[A-Za-z]+', word):
                                word = re.split('\d+', word)[1]
                            if re.match('[A-Za-z]+\d+', word):
                                word = re.split('\d+', word)[0]

                            if len(word) == 0 or len(
                                    word) == 1 or word == '' or word == ' ':
                                continue

                            word = ps.stem(word)

                            if extract_vocab:
                                vocab.add(word)
                            if extract_vectors:
                                doc_vector[self.vocab_index[word]] += 1

                    if extract_vectors:
                        X.append(doc_vector)
                        y.append(c)
        if extract_vocab:
            print(f'Vocab Size : {len(vocab)}')
            vocab_list = sorted(list(vocab))
            vocab_hash = dict.fromkeys(vocab_list, 0)
            vocab_index = {
                word: index
                for index, word in enumerate(vocab_list)
            }
            return vocab_index

        if extract_vectors:
            return (X, y)

    def get_query_vector(self, query_terms):
        ps = PorterStemmer()
        query_vector = pd.Series(self.data.T[0])
        query_terms = [ps.stem(word.lower()) for word in query_terms]
        for term in query_terms:
            if term in self.vocab_index.keys():
                if self.vocab_index[term] in self.idf.index:
                    query_vector.loc[self.vocab_index[term]] += 1
        for index in query_vector.index[query_vector > 0]:
            query_vector.loc[index] *= self.idf.loc[index]
        return query_vector

    def load_stop_words(self):
        stop_words = set()
        with open(self.stop_word_path, 'r') as stop_word_file:
            lines = stop_word_file.readlines()
            for line in lines:
                stop_words.add(line.split('\n')[0])
        return stop_words


def euclidian_distance(p1, p2):
    return np.linalg.norm(np.array(p2) - np.array(p1))


def cosine_similarity(p1, p2):
    return ((np.dot(p1, p2)) / (np.linalg.norm(p1) * np.linalg.norm(p2)))


def accuracy(y_test, pred):
    return len([1 for p, y in zip(pred, y_test) if p == y]) / len(pred) * 100


class KNNClassifier():
    def __init__(self, neighbors=3, distance_formula=euclidian_distance):
        self.distance_formula = distance_formula
        self.neighbors = neighbors
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, k=3):
        self.neighbors = k
        pred = []
        for index, test_row in X_test.iterrows():
            print(index)
            clear_output(wait=True)
            if self.distance_formula == euclidian_distance:
                pred.append(
                    self.X_train.apply(
                        (lambda row: self.distance_formula(row, test_row)),
                        axis=1).sort_values(ascending=True))
            else:
                pred.append(
                    self.X_train.apply(
                        (lambda row: self.distance_formula(row, test_row)),
                        axis=1).sort_values(ascending=False))

        new_pred = [x[:self.neighbors] for x in pred]
        label_pred = []
        for indexes in new_pred:
            labels = []
            #     print(indexes)
            for index, value in indexes.items():
                #         print(index)
                #         print(y[index])
                labels.append(self.y_train[index])
            label_pred.append(Counter(labels).most_common(1)[0][0])
        return label_pred


def build_knn_model(dataset, train_size, test_size, k, distance_formula, re_index):
    dataset_object = Dataset.objects.filter(dataset_name=dataset).latest()
    if re_index:
        dataset_object = Dataset.objects.filter(dataset_name=dataset).latest()
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        print(dataset_object)
        print(dataset)
        print(DATA_DIR)
        print(f'{DATA_DIR}\{STOPWORD_PATH}')
        dv = DocToVec(DOCUMENTS_PATH=f'{DATA_DIR}\{dataset_object.dataset_path}',
                      STOP_WORD_PATH=f'{DATA_DIR}\{STOPWORD_PATH}')

        mvs = ModelVectorSpace(status=True, )
        mvs.data = dv
        mvs.dataset = dataset_object
        mvs.save()
    else:
        mvs = ModelVectorSpace.objects.filter(dataset=dataset_object).latest()
        dv = mvs.data

    data = dv.data.copy()
    data['label'] = dv.y
    shuffled_data = data

    shuffled_data.sample(frac=1)
    train_size = train_size
    test_size = test_size

    train_data, test_data = shuffled_data.sample(
        frac=train_size), shuffled_data.sample(frac=test_size)
    X_train, y_train = train_data.loc[:,
                                      train_data.columns != 'label'], train_data['label']
    X_test, y_test = test_data.loc[:,
                                   test_data.columns != 'label'], test_data['label']

    if distance_formula == 'euclidian_distance':
        knn = KNNClassifier(distance_formula=cosine_similarity)
    else:
        knn = KNNClassifier(distance_formula=euclidian_distance)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test, k)

    knn_accuracy = accuracy(y_test, pred)

    knnM = KNNClasification(accuracy=knn_accuracy,
                            distance_formula=distance_formula, k=k, train_size=train_size, test_size=test_size)

    knnM.model = knn
    knnM.dataset = dataset_object
    knnM.vector_space = mvs
    knnM.save()
    return True


def get_knn_label(query, k, dataset):
    text = str(query)
    try:
        query_terms = [clean_word(word)
                       for word in re.split('[.\s\n\r,?!:;-]', text)]
    except ValueError as e:
        raise ValueError('Invalid Query Syntax')
    dataset_object = Dataset.objects.filter(dataset_name=dataset).latest()
    knnM = KNNClasification.objects.filter(dataset=dataset_object).latest()
    knn = knnM.model
    mvs = ModelVectorSpace.objects.filter(dataset=dataset_object).latest()
    dv = mvs.data
    ar = dv.get_query_vector(query_terms)
    label = knn.predict(pd.DataFrame([ar]), k)
    return label
