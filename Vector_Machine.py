from __future__ import print_function	# For Py2/3 compatibility
from nltk.stem import WordNetLemmatizer 
import math  
from nltk.stem.porter import *
import nltk
import json
import pandas as pd
import string
import re
#nltk.download('wordnet')
stemmer = PorterStemmer()
Ltoken = []
Lposting = []

lemmatizer = WordNetLemmatizer()
data = pd.read_json('./data/train.json')


######Vctor Generater#########     O(n1+n2)
def generate( ref_words, doc_freq, tokens, tf, i,n,file):  #Genrates Vector
    n1 = len(ref_words)               #Vector Length I equal to total no of terms
    n2 = len(tokens)                    #Terms length in current docs
    dot = 0
    j = 0
    k = 0
    
    vector = []
    while(k < n2 and j < n1 and tokens[k] != ''):   #Make vector untill terms in current doc ends 
        if(ref_words[j] != tokens[k]):
            file.write(str(0))
            file.write(',')
            vector.append(0)
            j += 1
        else:
            weight = (float(tf[k])/(len(tf)))*math.log10((n+1)/int(doc_freq[j]))
            dot += pow(weight, 2)                               #Calculation of tf^2 for getting Normalization
            j += 1
            k += 1
            file.write(str(weight))
            file.write(',')
            vector.append(weight)
          
    while(j < n1):                                         #cat remaining term weights as zero
        if(ref_words[j] == ''):
            j += 1
            continue
        file.write(str(0))
        file.write(',')
        vector.append(0)
        j += 1
    while(k < n2):
        print(tokens)
        if(tokens[k] == ''):
            k += 1
            continue
        k += 1
    file.write('\n')
    return dot,vector

#############Vector GEnerator  O(56)#####################################
def generate_vector(all_vocab,all_tf,doc_vocab,doc_tf):

    file = open('./Data/Vectors.csv','w')
    file.write("cuisine,")
    for i in range(0,len(all_vocab)):
        file.write(all_vocab[i])
        file.write(',')
    file.write('\n')
    n = len(doc_vocab)
    
    ref_words = all_vocab
    doc_freq = all_tf
    weights = []
    vectors = {}
    for i,value in doc_vocab.items():
        file.write(str(data.query('id=={}'.format(i))['cuisine'].iloc[0]))
        file.write(',')                                       #Open all docs and send all terms and terms in current doc for vector generation
        tokens = doc_vocab[i]
        tf = doc_tf[i]
        dot,vector = generate(ref_words, doc_freq, tokens, tf, i ,n,file)
        #vectors[str(i)] = vector
        weights.append(math.sqrt(dot))
    file.close()    

    file = open('./Data/Corpus.txt','w')
    file.write(",".join(map(str, all_vocab)))
    file.close()
    file = open('./Data/DocumentFrequency.txt','w')
    file.write(",".join(map(str, all_tf)))
    file.close()
    file = open('./Data/Weights.txt','w')
    file.write(",".join(map(str, weights)))
    file.close()
    #file  = open('./Data/vector.json','w')
    #json.dump(vectors,file)
    #file.close()
    
######################mergesort  O(n)###################################


def merge( arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m

    L = [0] * (n1)
    R = [0] * (n2)

# Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = arr[l + i]
    for j in range(0, n2):
        R[j] = arr[m + 1 + j]
# Merge the temp arrays back into arr[l..r]
    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = l     # Initial index of merged subarray

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def mergeSort( arr, l, r):
    if l < r:
        m = int((l+(r-1))/2)
        mergeSort(arr, l, m)
        mergeSort(arr, m+1, r)
        merge(arr, l, m, r)

def mergelists( x1, x2, ind, doc_freq):
    n1 = len(x1)
    n2 = len(x2)
    word = []
    freq_list = []
# Merge the temp arrays back into arr[l..r]
    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = 0     # Initial index of merged subarray

#word->dictionary, Plist->PostingList, posting-> <DocumentNumber>position of word
    while i < n1 and j < n2:
        if(x1[i] == ''):
            i += 1
            continue
        if(x2[j] == ''):
            j += 1
            continue

        if x1[i] < x2[j]:
            word.append(x1[i])  # Send Word To Dictionary
            freq_list.append(doc_freq[i])  # Append Previous Frequency
            i += 1
        elif x1[i] > x2[j]:  # Attach Document and Position since Second list's doc is smaller
            word.append(x2[j])
            freq_list.append(1)  # New Word Append 1 as frequency
            j += 1
        else:

            # Documewnts are same so join <docNo> and both postion Lists
            word.append(x1[i])
            freq_list.append(doc_freq[i]+1)  # Join Previous Frequency + 1
            i += 1
            j += 1

# Copy the remaining elements of list1, if there
# are any
    while i < n1:
        if(x1[i] == ''):
            i += 1
            continue
        word.append(x1[i])
        freq_list.append(doc_freq[i])
        i += 1

# Copy the remaining elements of list2, if there
# are any
    while j < n2:
        if(x2[j] == ''):
            j += 1
            continue
        word.append(x2[j])
        freq_list.append(1)
        j += 1
    return word, freq_list

def tokenizer():
    global Ltoken
    lem = WordNetLemmatizer()    
#Tokenize stem fold case of words
    doc_vocab = {}
    doc_tf = {}
    for i in range(data.shape[0]):
        cuisine = data.iloc[i].cuisine
        ingredient_string = data.iloc[i].ingredients
        id = data.iloc[i].id
        ingredient=[]
        for ing in ingredient_string:
            ingredient.extend(ing.split(' '))
    
        doc_set = set()
        for word in ingredient:
            terms = word.split(' ')
            final_term = ''
            for term in terms:
                term = term.translate(term.maketrans('','',string.punctuation))
                term = re.sub('[!@#$()]', '', term)
                term = term.lower()
                final_term+=lem.lemmatize(term)
            word=final_term
            if word not in ['(',')','[',']','.',',','']: 
            # and not word.isdigit():
                Ltoken.append(word)
        
    

        # Sorting and adding frequency of Tokens In file
        mergeSort(Ltoken, 0, len(Ltoken)-1)
        
       
        
    
        counter = 1
    # Write token and tf to if no preceding word is same as current
        words = []
        tf = []
        for l in range(0, len(Ltoken)-1):
            if Ltoken[l] != Ltoken[l+1]:
                words.append(Ltoken[l])
                tf.append(counter)
                counter = 1
            else:
                counter += 1  # preceding word is same increase tf
        words.append(Ltoken[len(Ltoken)-1])
        tf.append(counter)
        doc_vocab[str(id)] = words
        doc_tf[str(id)] = tf
        Ltoken.clear()
    
#Document as a BLOCK sorting done
    return doc_vocab,doc_tf

def Processor(doc_vocab,doc_tf):
    x1 = doc_vocab[list(doc_vocab.keys())[0]]
    doc_freq = [1]*len(x1)
    count = 1
    for i,value in doc_vocab.items():                                  #Takes tokens and respective term frequency from each token file and 
        if(count == 1):
            count+=1
            continue
        x2 = doc_vocab[i]
        x1, doc_freq = mergelists(x1, x2, i, doc_freq)
    all_vocab = []
         #Writing Doc Frequency
    for i in range(len(x1)):
        all_vocab.append(x1[i])
        all_tf = []
    for i in range(len(doc_freq)):
        all_tf.append(doc_freq[i])
    generate_vector(all_vocab,all_tf,doc_vocab,doc_tf)    



#############Merge sorted terms of each doc ############
def merge_doc( arr, pos, l, m, r):
    n1 = m - l + 1
    n2 = r - m

    L = [0] * (n1)
    R = [0] * (n2)
    LP = [0] * (n1)
    RP = [0] * (n2)
# Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = arr[l + i]
        LP[i] = pos[l + i]
    for j in range(0, n2):
        R[j] = arr[m + 1 + j]
        RP[j] = pos[m + 1 + j]

# Merge the temp arrays back into arr[l..r]
    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = l     # Initial index of merged subarray

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            pos[k] = LP[i]
            i += 1
        else:
            arr[k] = R[j]
            pos[k] = RP[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        pos[k] = LP[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        pos[k] = RP[j]
        j += 1
        k += 1

def sort_doc( arr, pos, l, r):        #######Sort tokenized terms of documents
    if l < r:
        m = int((l+(r-1))/2)
        sort_doc(arr, pos, l, m)
        sort_doc(arr, pos, m+1, r)
        merge_doc(arr, pos, l, m, r)

def parse_Query( query):
    stopwords = []
    s = open('./Data/Stopword-List.txt', 'r')
    stopdoc = s.read()
    w = ''
    for i in range(len(stopdoc)):  # Get Stopwords to remove from query
        if(stopdoc[i] == '\n'):
            if(w != ''):
                stopwords.append(w)
            w = ''
        elif stopdoc[i] != ' ':
            w += stopdoc[i]
    s.close()
    parsed = []
    w = ''
    for i in range(len(query)):  # parse Query
        if(query[i] == ' '):
            if(w not in ['', ",", ' ']):
                parsed.append(w)
            w = ''
        elif query[i] not in [' ', "'", '"', '']:
            w += query[i]
    s.close()
    if w not in ['', ",", ' ']:
        parsed.append(w)
    for i in range(len(parsed)-1, -1, -1):
        if(parsed[i] in stopwords):   #######Removing Stop Words #######
            parsed.pop(i)
        else:
            parsed[i] = parsed[i].lower()
            parsed[i] = lemmatizer.lemmatize(parsed[i])
    print("Unsorted Query =  {}".format(parsed))
    parsed.sort()
    print("Sorted Query = {}".format(parsed))
    return parsed

def binarySearch( arr, l, r, x):   ###########Search Words IN query through Binary searcg    Log(n)

    while l <= r:

        mid = int(l + (r - l)/2)

    # Check if x is present at mid
        if arr[mid] == x:
            return mid

    # If x is greater, ignore left half
        elif arr[mid] < x:
            l = mid + 1

    # If x is smaller, ignore right half
        else:
            r = mid - 1

    # If we reach here, then the element was not present
    return -1

def process_Query( parsed,corpus):
    tokens = corpus
    count = 1
    new_parsed = []
    query_tf = []
    for i in range(len(parsed)-1):                  #if terns in query is redundant just add 1 to count
        if(parsed[i] == parsed[i+1]):
            count += 1
        else:
            new_parsed.append(parsed[i])            #append single term and tf 
            query_tf.append(count)
            count = 1
    new_parsed.append(parsed[len(parsed)-1])
    query_tf.append(count)
    result = []
    for i in range(len(new_parsed)):               #Find occurence of query terms in corpus
        result.append(binarySearch(
            tokens, 0, len(tokens)-1, parsed[i]))

    return result, query_tf, new_parsed


def fetch_docs( parsed, query_tf, results,all_vectors,weights,documentFreq):
    
    #print(results)
    query_vector = []
    doc_freq = documentFreq
    ########
    weight = []
    print("Genrating Vector From Query")
    for i in range(len(results)):     #Making Vector from query term tf
        print("Document Frequency of {} = {}".format(
            parsed[i], doc_freq[results[i]]))
        query_vector.append((math.log10(
            int(doc_freq[results[i]]))/56)*query_tf[i])
        print("{}-->{}".format(parsed[i],(math.log10(int(doc_freq[results[i]]))/56)*query_tf[i]))
    print("------------")
    print("Query Vector")
    print(query_vector)
    print("------------")
    normal_vectors = weights
    vectors = all_vectors
    
    for i in range(56):                                                         #Vector addition of term in query with terms in each doc. Only weights 
        vector = vectors[str(i)]
        ans = 0
        a = 0
        print("Vector For Doc {}".format(i))
        print("<",end='')
        for j in range(len(query_vector)):
            print("{},".format(vector[results[j]]),end=' ')
            ans = ans+query_vector[j]*float(vector[results[j]])
            a += pow(query_vector[j], 2)
        print(">")
        b = float(normal_vectors[i])
        try:
            rslt = round(ans/(math.sqrt(a)*b), 10)
            print("<A><B>/|A||B| = {}/ |{}||{}|".format(ans,math.sqrt(a),b))
        except:
            rslt = 0
        weight.append(rslt)

    return weight


doc_vocab,doc_freq = tokenizer()

# file = open('./Data/vocab.txt','w')
# json.dump(doc_vocab,file)
# file.close()
# file = open('./Data/termFreq.txt','w')
# json.dump(doc_freq,file)
# file.close()

# file = open('./Data/vocab.txt','r')
# doc_vocab=json.load(file)
# file.close()
# file = open('./Data/termFreq.txt','r')
# doc_freq=json.load(file)
# file.close()
Processor(doc_vocab,doc_freq)