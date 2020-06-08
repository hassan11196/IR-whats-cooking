from classification.models import Dataset, ModelVectorSpace, ClassificationMlModel
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib
from collections import Counter
from nltk.stem import PorterStemmer
import random
from copy import deepcopy
import numpy as np
from IPython.display import clear_output
from IPython.core.interactiveshell import InteractiveShell
import pickle
import os
import string
import pandas as pd
import re
import json
import scipy


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


class JSONDocToVec(object):
    def __init__(self, DOCUMENTS_PATH, STOP_WORD_PATH):
        self.ingredients = set()
        self.doc_index = {}
        self.documents_path = DOCUMENTS_PATH
        self.stop_word_path = STOP_WORD_PATH
        self.stop_words = self.load_stop_words()
        self.Xindex = []
        self.vocab_index = self.file_extraction_wrapper(extract_vocab=True)
        vectors = self.file_extraction_wrapper(extract_vectors=True)
        self.files = {}

#         self.X = self.vectors[0]
        self.y = vectors[1]

        data = pd.DataFrame(vectors[0])
        # # Feature Selection
        # Drop Features with Df < 3
        data.drop([
            col for col, val in data.sum().iteritems() if int(val) <= 3
        ], axis=1, inplace=True)
        data.mul(data.sum().apply(lambda df: np.log10(data.shape[0] / (df + 1))),
                 axis=1)
        self.data = scipy.sparse.csr_matrix(data.values)
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
            Xindex = []
        doc_count = 0
        # Printable characters are
        # 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
        # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c
        ps = PorterStemmer()
        json_files = next(os.walk(os.path.join(self.documents_path)))[2]
        print('dir : ',  list(os.walk(os.path.join(self.documents_path))))
        print(json_files)
        for jfile in json_files:
            #             docs_in_c = next(os.walk(os.path.join(self.documents_path, c)))[2]
            if jfile.startswith('test'):
                continue
            print(jfile)
            print(self.documents_path)
            print('filepath : ', (os.path.join(self.documents_path, jfile)))

            with open(os.path.join(self.documents_path, jfile),
                      'r') as file1:
                rows = json.load(file1)

                for doc in rows:

                    if extract_vectors:
                        doc_vector = np.zeros((len(self.vocab_index)))
#                         doc_name = os.path.join(self.documents_path, c, doc)
                        self.doc_index[doc_count] = doc['id']
                        doc_count += 1

#                         symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
#                         for i in symbols:
#                             line = line.replace(i, ' ')
                    for word in doc['ingredients']:

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
                        if extract_vocab:
                            self.ingredients.add(word)
                        word = ps.stem(word)
#                         print(word)
                        if extract_vocab:

                            vocab.add(word)
                        if extract_vectors:
                            doc_vector[self.vocab_index[word]] += 1

                    if extract_vectors:
                        Xindex.append(doc['id'])
                        X.append(doc_vector)
                        if jfile.startswith('test'):
                            y.append(None)
                        else:
                            y.append(doc['cuisine'])
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
            self.Xindex = Xindex
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


def build_ml_model(dataset, train_size, test_size, ml_model_type, re_index, **kwargs):
    dataset_object, created = Dataset.objects.get_or_create(
        dataset_name=dataset)

    if created:
        dataset_object.dataset_path = 'whats-cooking'
        dataset_object.save()
    if True:
        dataset_object = Dataset.objects.filter(dataset_name=dataset).latest()
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        print(dataset_object)
        print(dataset)
        print(DATA_DIR)
        print(f'{DATA_DIR}\{STOPWORD_PATH}')
        dv = JSONDocToVec(DOCUMENTS_PATH=f'{DATA_DIR}\{dataset_object.dataset_path}',
                          STOP_WORD_PATH=f'{DATA_DIR}\{STOPWORD_PATH}')

        mvs = ModelVectorSpace(status=True, )
        mvs.data = dv
        mvs.dataset = dataset_object
        mvs.save()
    else:
        mvs = ModelVectorSpace.objects.filter(dataset=dataset_object).latest()
        dv = mvs.data
    print('Vector Space Created')
    data = pd.DataFrame(dv.data.toarray())
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

    # if distance_formula == 'euclidian_distance':
    #     knn = KNNClassifier(distance_formula=cosine_similarity)
    # else:
    #     knn = KNNClassifier(distance_formula=euclidian_distance)
    print('Train Test Split Completed')
    if ml_model_type == 'RandomForestClassifier':
        ml_model = RandomForestClassifier(n_jobs=-1)

    ml_model.fit(X_train, y_train)
    pred = ml_model.predict(X_test)
    print('Model Fitting Completed')
    ml_model_accuracy = accuracy_score(y_test, pred)
    print('Accuracy : ', ml_model_accuracy)
    cmm = ClassificationMlModel(accuracy=ml_model_accuracy,
                                ml_model_type=ml_model_type, train_size=train_size, test_size=test_size)
    print('Creating DB Model ')
    cmm.model = ml_model
    print('1')
    cmm.dataset = dataset_object
    print('2')
    cmm.vector_space = mvs
    print('3')
    cmm.save()
    print('Model Saved Completed')
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
