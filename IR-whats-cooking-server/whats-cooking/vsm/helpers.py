import pickle
import os
import string

import re
import math
import pprint

import numpy as np
import copy
from copy import deepcopy
from nltk.stem import WordNetLemmatizer

from vsm.models import VectorSpaceModel
FILE_PATH = os.path.dirname(__file__) + '../../data/' + 'Trump Speechs/speech_'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lem = WordNetLemmatizer()
# Remove Punctuation
def remove_punctuation(word):
    return word.translate(word.maketrans('','',string.punctuation))

# Clean Query Term
def clean_word(word):
    # Case Folding
    word = word.lower()
     # Filter non-ASCII characters
    word = ''.join(filter(lambda x: x in printable, word))
#     print(word)
    # Remove Punctuations
    if word != '(' and word != ')':
        word = remove_punctuation(word)
#     print(word)
    if re.match('\d+[A-Za-z]+',word):
        word = re.split('\d+',word)[1]
    if re.match('[A-Za-z]+\d+',word):
        word = re.split('\d+',word)[0]
#     print(word)
    word = ps.stem(word)
#     print(word)
    return word
class VectorSpace(object):
    def tf_natural(self, tf, doc):
        #return tf
        return tf / sum(doc.values())
    def tf_logarithm(self, tf, doc):
    
        return (0 if tf==0 else 1 + math.log10(tf))
    def tf_augmented(self, tf, doc):
        #return tf
        return (0.5 + ((0.5 * tf)/(self.find_max_tf(doc))))
    def tf_boolean(self, tf, doc):
        #return tf
        return (1 if tf > 0 else 0)

    def idf_no(self, term):
        return 1
    def idf_idf(self, term):
        return math.log10((len(self.docs.keys()) / (len(self.index[term]))) + 1 )
    def norm_cosine(doc):
        return 1 / (sum([tf**2 for tf in doc.values()]) + 1)
    def norm_no(self, doc):
        return 1  
    def __init__(self, tf_func = 'natural', idf_func = 'idf', norm_func='none'):
        self.tf_functions = {
            'natural': self.tf_natural,
            'logarithm': self.tf_logarithm ,
            'augmented': self.tf_augmented ,
            'boolean' : self.tf_boolean ,
            'log_ave': self.tf_log_ave   
        }
        self.idf_functions = {
            'no': self.idf_no,
            'idf':self.idf_idf,
            'prob_idf' : self.prod_idf
        }
        self.normailization_functions = {
            'none' : self.norm_no,
            'cosine' : self.norm_cosine,
#             'pivoted_unique' : lambda 
        }
        self.tf_func = self.tf_functions[tf_func]
        self.idf_func = self.idf_functions[idf_func]
        self.norm_func = self.normailization_functions[norm_func]
        
        self.vocab = {}
        self.vocab_idf = {}
        self.docs = {}
        self.docs_char_length = {}
        self.occurrance = {}
        self.occurrance2 = {}
        self.cdocs = {}
        self.index = {}
        self.func_strings = {'tf_func' : tf_func, 'idf_func':idf_func, 'norm_func':norm_func}
    def tf_log_ave(self, tf, doc):
#         print(tf)
        return ( (1+math.log10((tf*sum(doc.values()))+1)) / (1 + math.log10(self.find_avg_tf(doc))))
    def prod_idf(self, term):
#         print('prob idf')
#         print(math.log10((len(self.docs.keys()) - (len(self.index[term])) + 1)/(len(self.index[term]))))
        if 0 > math.log10((len(self.docs.keys()) - (len(self.index[term])) + 1)/(len(self.index[term]))):
            return 0;
        else:
            return math.log10((len(self.docs.keys()) - (len(self.index[term])) + 1)/(len(self.index[term])))    
    
    
    def find_avg_tf(self, doc):
        csum = 0
        cnt = 0
#         print(f'Doc sum : {sum(doc.values()) }')
#         print(sum(doc.values()) / len(doc.keys()))
        return (sum(doc.values()) / len(doc.keys()))
    
    def find_max_tf(self, doc):
        max_tf = 0
        max_term = None
        for term, tf in doc.items():
            if tf > max_tf:
                max_tf = tf
                max_term = term
        return max_tf

    
    
    def create_doc(self, docId):
        self.docs[docId] = dict.fromkeys(self.vocab, 0)
        
    def add_term(self, term, docId, position):
        if term not in self.vocab.keys():
            self.vocab[term] = 1
            for Id, docList in self.docs.items():
                self.docs[Id][term] = 0
        else:
            self.vocab[term]+=1
        
        if term in self.docs[docId].keys():
            self.docs[docId][term] += 1
        else:
            self.docs[docId][term] = 1

        # if term not in self.occurrance.keys():
        #     self.occurrance[term] = {}
        #     self.occurrance[term][docId] = []
        #     self.occurrance[term][docId].append(position)
        # else:
        #     if docId not in self.occurrance[term].keys():
        #         self.occurrance[term][docId] = []
        #         self.occurrance[term][docId].append(position)
        #     else:
        #         self.occurrance[term][docId].append(position)  

        if docId not in self.occurrance2.keys():
            self.occurrance2[docId] = {}
            self.occurrance2[docId][term] = []
            self.occurrance2[docId][term].append(position)
        else:
            if term not in self.occurrance2[docId].keys():
                self.occurrance2[docId][term] = []
                self.occurrance2[docId][term].append(position)
            else:
                self.occurrance2[docId][term].append(position)



        if term in self.index.keys():
            self.index[term].add(docId)
        else:
            self.index[term] = set()
            self.index[term].add(docId)
#         if docId not in self.occurrance.keys():
#             self.occurrance[docId] = []
#         self.occurrance[docId].append(position)
    
    def get_query_vector(self, query_terms):
        query_vector_hash = dict.fromkeys(self.vocab, 0)
#         print(query_vector_hash)
        query_terms = [lem.lemmatize(word.lower()) for word in query_terms]
        print(query_terms)
        for term in query_terms:
            if term in query_vector_hash.keys():
                query_vector_hash[term] += 1

        
        words_in_query = len(query_terms)
        tf = dict.fromkeys(self.vocab_idf, 0) 
        tf_idf = dict.fromkeys(self.vocab_idf, 0)
        # norm = self.norm_func(query_vector_hash)
        for term,term_cnt in query_vector_hash.items():
            if term_cnt <= 0:
                continue
#                 print(term)
#                 print(term_cnt)
#                 tf[term] = term_cnt / words_in_query
#                 idf[term] = len(self.docs.keys()) / (self.vocab[term])

            tf[term] = self.tf_func( term_cnt,query_vector_hash )
            tf_idf[term] = tf[term] *  self.vocab_idf[term]
            print(f'tf_idf {term}: {tf_idf[term]}')
        new_query_vector_hash = {
            'tf': tf,
            'idf':self.vocab_idf,
            'tf_idf':tf_idf
        }
    
    
        return new_query_vector_hash
    
    def dot_product(self, v_hash_1, v_hash_2):
#         print(len(v_hash_1))
#         print(len(v_hash_2))
#         print('vhash')
#         print(v_hash_2)
        return sum([v_hash_2[x]*y for x,y in v_hash_1.items()])
        
    def get_magnitude(self, v_hash):
        mag = sum([x**2 for x in v_hash.values()]) ** 0.5 
        if mag == 0:
            return 1
        return mag
    
    
    def get_cosine_sim(self, v_hash_x, v_hash_y):
        return (self.dot_product(v_hash_x,v_hash_y)) / (self.get_magnitude(v_hash_x) * self.get_magnitude(v_hash_y)) 
            
    def get_ranking(self, query_vector_hash):
        ranked_docs = []
        for docId,doc_vector_hash in self.cdocs.items():
            cosine_sim = self.get_cosine_sim(doc_vector_hash['tf_idf'], query_vector_hash['tf_idf'])
            ranked_docs.append((docId, cosine_sim))
        ranked_docs = sorted(ranked_docs, key=lambda x:x[1])
        return ranked_docs
    def calculate_tf_idf(self):
        cdocs = {}
        self.vocab_idf = dict.fromkeys(self.vocab, 0)
        for term, term_cnt in self.vocab.items():
            self.vocab_idf[term] = self.idf_func(term)
            
        for docId,doc in self.docs.items():
#             print(doc.values())
            norm = self.norm_func(doc)
            words_in_d = sum(doc.values())
            tf = {}
        
            tf_idf = {}
            for term,term_cnt in doc.items():
#                 tf[term] = term_cnt / words_in_d
#                 idf[term] = len(self.docs.keys()) / (self.vocab[term])
                tf[term] =self.tf_func(term_cnt, doc)
                tf_idf[term] = tf[term] * self.vocab_idf[term] * norm
            cdocs[docId] = {
                'tf': tf,
                'tf_idf':tf_idf,
                'total_words' : words_in_d
            }
        self.cdocs = cdocs
        return self.cdocs


def build_index(tf_func='natural',idf_func='idf', norm_func='none'):
    BASE_URL = os.path.join(BASE_DIR, 'whats-cooking/static')
    print(BASE_URL)
    
    vocab = set()
    doc_contents = []
    vector_space = VectorSpace(tf_func=tf_func,idf_func=idf_func, norm_func=norm_func)
    printable = set(string.printable) 
    # Printable characters are
    # 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
    # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c


    lem = WordNetLemmatizer() 
    stop_words = set()
    with open(BASE_URL+'/Stopword-List.txt', 'r') as stop_word_file:
        lines = stop_word_file.readlines()
        for line in lines:
            stop_words.add(line.split('\n')[0])
        stop_words.remove('')
    print(stop_words)

    for file_number in range(0, 56):
        vector_space.create_doc(file_number)
        with open(BASE_URL +f'/speech_{file_number}.txt', 'r') as file1:
            lines = file1.readlines()
            print(f'File Number : speech_{file_number}.txt' )
            print(lines[0])
            position = {'doc':file_number,'row':0, 'col':0, 'token_no':0}

            for line_no,line in enumerate(lines):
                doc_set = set()
                # Skip Heading Line
                if line_no == 0:
                    continue
                # split words at . , whitespace ? ! : ;
                position['row'] = line_no 
                position['col'] = 0
                for word in re.split("[.\s,?!:;-]", line):
                    position['col'] += len(word) + 1
                    position['token_no'] += 1
                    # Case Folding
                    word = word.lower()
                    
                    # Filter non-ASCII characters
                    word = ''.join(filter(lambda x: x in printable, word))
                    
                    # Remove Punctuations
                    word = remove_punctuation(word)
                    
                    if re.match('\d+[A-Za-z]+',word):
                        word = re.split('\d+',word)[1]
                    if re.match('[A-Za-z]+\d+',word):
                        word = re.split('\d+',word)[0]
                    
                    if len(word) == 0 or len(word) == 1 or word == '' or word == ' ':
                        continue
                    if word in stop_words:
                        continue

                    word = lem.lemmatize(word)
                        
                    vocab.add(word)
                    
                    doc_set.add(word)
                    
                    vector_space.add_term(word, file_number, deepcopy(position))
            doc_contents.append(doc_set)
            print('*', end='')
    doc_term_tf_idf = vector_space.calculate_tf_idf()
    ii = VectorSpaceModel()
    ii.status = True
    ii.tf_func = tf_func
    ii.idf_func = idf_func
    ii.norm_func = norm_func
    ii.data = vector_space
    ii.save()
    return True


def get_vector_query(query, alpha=0.0005):
    text = str(query)
    try:
        query_terms = text.split(' ')
    except ValueError as e:
        raise ValueError('Invalid Vector Query Syntax')
    lem = WordNetLemmatizer()
    query_terms = [lem.lemmatize(x.lower()) for x in query_terms]

    vsm_model_obj = VectorSpaceModel.objects.latest('id')
    vector_space = vsm_model_obj.data
    print('vsm')
    result = [] 
    ranked = vector_space.get_ranking(vector_space.get_query_vector(query_terms))
    ranked.reverse()
    result = [(x,y) for x,y in ranked if y > alpha]
    print(result)

    occurance = {}
    for docId in vector_space.occurrance2.keys():
        for query in query_terms: 
            if query in vector_space.occurrance2[docId].keys():
                if docId in occurance.keys():
                    occurance[docId].append(vector_space.occurrance2[docId][query])
                else:
                    occurance[docId] = []
                    occurance[docId].append(vector_space.occurrance2[docId][query])
    result = {
        'doc_ids' : sorted(result,key= lambda x:x[1], reverse=True),
        'occurrance' : occurance
    }
    return result