
import os
import pickle
from nltk.stem.porter import *


stemmer = PorterStemmer()

doc_file = open('doc_desc.pickle', 'rb')

with open('idf_desc.pickle', 'rb') as idf_file:
    idf = pickle.load(idf_file)
    idf = idf[0]

docs = []
i = 0
while(i < 13574):
    docs.append(pickle.load(doc_file))
    i = i+1

def ret_docs():
    return docs

def ret_idf():
    return idf             