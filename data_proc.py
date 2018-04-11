import os
import nltk
from nltk.stem.porter import *
import math
import pickle
import re
from nltk.corpus import stopwords

def sum_of_squares(dic):
    ''' Returns sum of squares of values 
        associated with every key in the dictionary(dict)
    '''
    return sum(value**2 for key, value in dic.items() )

stemmer = PorterStemmer()   # Porter Stemmer object

idf = {}                    # IDF dictionary which have words and it's IDF as (key, value) pairs

data_path = './Documents'
sample_path = './sample'

desc = open('doc_desc.pickle', 'wb')    # Creates doc_desc.pickle file to store the descriptions of every document
f_desc = open('idf_desc.pickle', 'wb')  # Creates idf_desc.pickle file to store the description of idf  
n_files = 0

for filename in os.listdir(data_path):
    with open(data_path+'/'+filename, 'r') as file:
        raw_data = file.read().split('\n')
        
        link = raw_data[0]                                                  #
        raw_data = [word.lower() for word in raw_data]                      #
        data = []                                                           #   Data Pre-processing
        for word in raw_data:                                               #       -Tokenized and removed all the special characters and numeric data
            data.append(re.sub('[^a-z]+', ' ', word))                       #       -Converted data into lower case
                                                                            #       -Removed stop words
        tokens = nltk.word_tokenize(data[2])                                #       -Stemmed all the words using stemmer object
        stop_words = set(stopwords.words('english'))                        #
        filtered_tokens = [w for w in tokens if not w in stop_words]        #
                                                                            #
        output=[stemmer.stem(word) for word in filtered_tokens]             
        dtf = {}
        for term in output:                                                 #
            value = output.count(term)                                      #
            value = 1 + math.log10(value)                                   #   Document Term Frequency
            dtf.update({term:value})                                        #       -dtf is calculated by iterating through every word and updating the old values
                                                                            #       -All dtf values are normalized
        for key, value in dtf.items():                                      #
            val = idf.get(key)                                              #
            if val is None:                                                 #
                idf.update({key:1})                                         #
            else:
                idf.update({key:val+1})
        
        dnom = math.sqrt(sum_of_squares(dtf))
        normalized_dtf = {}
        
        for key, value in dtf.items():                                      
            normalized_dtf.update({key:value/dnom})
        
        doc = []                                                            #
        doc.append(filename)                                                #   Appended properties of the document     
        doc.append(link)                                                    #
        doc.append(normalized_dtf)                                          #
        
        pickle.dump(doc, desc)                                              #        
        n_files = n_files + 1                                               #   Dumping into desc
        print(n_files)                                                      #
        
for key, value in idf.items():
    idf.update({key: math.log10(n_files/value)})

# =====================================================================
idf_list = [idf]
# =====================================================================
pickle.dump(idf_list, f_desc)                                               #
f_desc.close()                                                              #   Dumping into idf pickle file
desc.close()                                                                #

print('Finished Successfully ')
        
        
        
        
        
        
        
            
        

        
    
    