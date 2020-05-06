import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import string
from nltk.stem import WordNetLemmatizer 
import re
import math
import pprint
import numpy as np
import copy
from copy import deepcopy
import itertools  


data = pd.read_json('./Data/train.json')
#data = data.sample(frac=0.01,random_state=1)
#print(data.head())
#del data['id']
print(data.groupby('cuisine').size())

def remove_punctuation(word):
    return word.translate(word.maketrans('','',string.punctuation))

class VectorSpaceModel(object):
    
    def tf_natural(self, tf, doc):
    #return tf
        return tf / sum(doc.values())
    def idf_no(self, term):
        return 1
    def idf_idf(self, term):
        return math.log10((len(self.docs.keys()) / (len(self.index[term]))) + 1 )
    def norm_cosine(self,doc):
        return 1 / (sum([tf**2 for tf in doc.values()]) + 1)
    def norm_no(self, doc):
        return 1  
    def __init__(self, tf_func = 'natural', idf_func = 'idf', norm_func='none'):
        self.tf_functions = {
            'natural': self.tf_natural,
            # 'logarithm': lambda tf, doc : (0 if tf==0 else 1 + math.log10(tf)),
            # 'augmented': lambda tf, doc : (0.5 + ((0.5 * tf)/(self.find_max_tf(doc)))),
            # 'boolean' : lambda tf, doc : (1 if tf > 0 else 0),
            # 'log_ave': self.tf_log_ave

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
        def tf_log_ave(self, tf, doc):
    #         print(tf)
            return ( (1+math.log10((tf*sum(doc.values()))+1)) / (1 + math.log10(self.find_avg_tf(doc))))
    def prod_idf(self, term):
#         print('prob idf')
#         print(math.log10((len(self.docs.keys()) - (len(self.index[term])) + 1)/(len(self.index[term]))))
        if 0 > math.log10((len(self.docs.keys()) - (len(self.index[term])) + 1)/(len(self.index[term]))):
            return 0
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
        
    def add_term(self, term, docId):
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
        
                
        if term in self.index.keys():
            self.index[term].add(docId)
        else:
            self.index[term] = set()
            self.index[term].add(docId)
#         if docId not in self.occurrance.keys():
#             self.occurrance[docId] = []
#         self.occurrance[docId].append(position)
    
   

    def calculate_tf_idf(self):
        cdocs = {}
        file = open('vectors.csv','w')
        file.write("cuisine,")
        self.vocab_idf = dict.fromkeys(self.vocab, 0)
        for term, term_cnt in self.vocab.items():
            self.vocab_idf[term] = self.idf_func(term)
            file.write(str(term))
            file.write(',')
        file.write('\n')
        
        for docId,doc in self.docs.items():
            #print("{},{}".format(docId,doc))
#             print(doc.values())
            words_in_d = sum(doc.values())
            tf = {}
        
            tf_idf = {}
            file.write(str(data.query('id=={}'.format(docId))['cuisine'].iloc[0]))
            file.write(',')
            for term,term_cnt in doc.items():
#                 tf[term] = term_cnt / words_in_d
#                 idf[term] = len(self.docs.keys()) / (self.vocab[term])
                tf[term] =self.tf_func(term_cnt, doc)
                #tf_idf[term] = tf[term] * self.vocab_idf[term]
                file.write(str(tf[term] * self.vocab_idf[term]))
                file.write(',')
            file.write('\n')
            #cdocs[docId] = {
                
             #   'tf_idf':tf_idf,
            
            #}
        file.close()
        return self.cdocs
    
vocab = set()
doc_contents = []

vector_space = VectorSpaceModel(tf_func='natural',idf_func='idf')
printable = set(string.printable) 
raw_data = []
# Printable characters are
# 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
# !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c


lem = WordNetLemmatizer() 

stop_words = set()
#import nltk
#nltk.download('wordnet')


for i in range(data.shape[0]):
    cuisine = data.iloc[i].cuisine
    ingredient_string = data.iloc[i].ingredients
    id = data.iloc[i].id
    vector_space.create_doc(id)
    ingredient=[]
    for ing in ingredient_string:
      ingredient.extend(ing.split(' '))
    
    
    # print(ingredient)
    # break
    doc_set = set()
    for word in ingredient:
        terms = word.split(' ')
        final_term = ''
        for term in terms:
            final_term+=lem.lemmatize(term)
        word=final_term
        vocab.add(word)
    
        doc_set.add(word)
        # print(word)
        vector_space.add_term(word, id)
            
    doc_contents.append(doc_set)
    # print('*', end='')
doc_term_tf_idf = vector_space.calculate_tf_idf()
print('Done')
print('Total Vocabulary Size ')
print(len(vector_space.index.keys()))
print('Total Number of Documents ')
print(len(vector_space.docs))