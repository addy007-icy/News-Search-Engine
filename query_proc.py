import os
import pickle
from nltk.stem.porter import *
import nltk
import math
from operator import itemgetter
from nltk.corpus import stopwords
import timeit
import pickle_loader

stemmer = PorterStemmer()
    
idf = pickle_loader.ret_idf()                   #   Read pickle files
docs = pickle_loader.ret_docs()                 #   -idf, docs

def ask_query(query):
    ''' Function returns top 10 results 
        relevant to the passed free text query as an argument
    '''
    
    start = timeit.default_timer()                                          #
    tokens = nltk.word_tokenize(query.lower())                              #   Pre-processing of query
    stop_words = set(stopwords.words('english'))                            #       -Tokenized
    filtered_tokens = [w for w in tokens if not w in stop_words]            #       -Removed stop words
    output=[stemmer.stem(word) for word in filtered_tokens]                 #       -Stemmed the obtained query
    qtf = {}
    
    for term in output:                                                     
        value = output.count(term)
        value = 1 + math.log10(value)                                       #
        qtf.update({term:value})                                            #
                                                                            #        
    query_vector = {}                                                       #   Query Term Frequency
                                                                            #       -Variable -> qtf
    for key, value in qtf.items():                                          #                
        idf_val = idf.get(key)                                              #   Query Vector     
        if idf_val is None:                                                 #       -Used idf calculated in data_proc.py to compute the query vector
            idf_val = 0     
        query_vector.update({key:idf_val*value})
    
    score = []
    
    for doc in docs:
        filename = doc[0]
        link = doc[1]                                                       #            
        n_dtf = doc[2]                                                      #
        doc_score = 0                                                       #
        for key, value in query_vector.items():                             #   Dot product of query vector and document vector to calculate document score for each document in the corpus
            dtf = n_dtf.get(key)                                            #
            if dtf is None:                                                 #
                dtf = 0                                                     #
            doc_score = dtf*value + doc_score                       
        
        score.append((filename, doc_score, link))
    
    n_results = 10                                                          #
    results = sorted(score, key = itemgetter(1))                            #   Retrieved top 10 results 
    final_results = results[-11:-1]                                         #        
    
    stop = timeit.default_timer()
    time = stop - start
    
    return final_results, time
    

def more_results(query):
    start = timeit.default_timer()
    sample= query
    tokens = nltk.word_tokenize(sample.lower()) 
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    output=[stemmer.stem(word) for word in filtered_tokens]
    results = []
    
    for t in output:
        removal= [t]
        new_tokens = [w for w in output if not w in removal] 
        new_query=""
        for n in new_tokens:                                    #
            new_query= new_query+" "+n                          #   Created new queries
        new_results, new_time = ask_query(new_query)            #

        for r in new_results:
            results.append(r)
    sorted(results, key = itemgetter(1))
    stop = timeit.default_timer()
    time = stop - start
    return results[-11:-1], time

