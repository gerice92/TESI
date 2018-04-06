from __future__ import division
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support as pr
import csv #package to work with tsv files
import re #regex
import numpy as np

#FUNCTION TO GET THE NUMBER OF SYLLABES IN WORD
def syll_counter(word):
    #a function to count the number of syllables
    syll = lambda w:len(''.join(c if c in"aeiouy"else' 'for c in w.rstrip('e')).split())
    num_syl=syll(word)
    return num_syl

#GET THE LENGTH OF A TEXT
def length(text):
    length = len(text)
    return length

def get_window(word, sentence, start):

    #Split sentence by whitespaces to get all tokens
    tokens = sentence.split()
    start=int(start) #start of word

    #we return a 2-window around the word
    w_2 = "" 
    w_1 = ""
    w1 = ""
    w2 = ""

    #count return the number of times that word appear in a list
    num = tokens.count(word)

     #index of the word at tokens
    index=-1

    if num > 0:
        if num == 1 or start == 0:
            index = tokens.index(word)
        else:
            sentence1 = sentence[0:start-1]
            tokens1=sentence1.split(' ')
            index=len(tokens1) 
            if (index>-1 and tokens[index]!=word):
                index=-1
                #print('Problems with',sentence,word)

    if (index-1>=0):
        w_1=tokens[index-1] 
        if (index-2>=0):
            w_2=tokens[index-2]
    if (index+1<=len(tokens)-1):
        w1=tokens[index+1]
        if (index+2<=len(tokens)-1):
            w2=tokens[index+2]
                            
    return w_2,w_1,w1,w2

#LOAD GRAMS
def load_dic(path):
    dic={}
    f=open(path,'r')
    for line in f:
        line=line.strip()
        pos=line.rfind(' ')
        key=line[0:pos]
        freq=line[pos+1:len(line)]
        dic[key]=int(freq)
    
    f.close()
    return dic

'''GET PROBABILITY OF OCURRENCES OF WORD
    ngram: gram to evaluate
    dic: dictionary to use
    size: max value of frequency in dic
'''
def getProbability(ngram,dic,size):
    ngram=ngram.lower() #lowcase
    
    ngram=ngram.strip()
    prob=0
    try:
        prob=dic[ngram]
        prob=prob/size
    except:
        pass

    #prob=prob*SCALED
    return prob

#GET NUMBER OF EXAMPLES OF DATASET
def getLines(path):
    f=open(path,'r') 
    numExamples = sum(1 for line in f)  # fileObject is your csv.reader
    f.close()
    return numExamples

#CREATE OUTPUT MATRIX
def getMatrix(path):
    
    numExamples = getLines(path)

    numFeatures = 9 #number of features
    matrix = np.empty(shape = [numExamples, numFeatures])

    #Load grams
    #Unigrams
    dic_path = 'LangModels/wiki_1.wngram'
    unigrams = load_dic(dic_path)
    totalUnis = sum(unigrams.itervalues())
    maxValue = max(unigrams.itervalues())

    #Bigrams
    dic_path = './LangModels/wiki_2.wngram'
    bigrams = load_dic(dic_path)
    totalBis = sum(bigrams.itervalues())

    #Trigrams
    dic_path = './LangModels/wiki_3.wngram'
    trigrams = load_dic(dic_path)
    totalTris = sum(trigrams.itervalues())

    #Dataset path    
    tsvin = open (path, "rb")
    tsvin = csv.reader(tsvin, delimiter="\t")

    indexRow = 0 #Index of vectors in matrix
    for row in tsvin:
        ide = row[0] #paragraph ID
        sentence = row[1] #sentence
        start = row[2] #initial position of word
        end = row[3] #final position of word
        word = row[4] #word to classify
        class_word = row[9] #clase: 1 o 0
        probability = row[10] #(n 1/ n anotadores)  
    
        len_word=len(word)
        num_syl=syll_counter(word)
        len_sen=len(sentence)

        #Get Window
        w_2,w_1,w1,w2=get_window(word,sentence,start)

        #Get grams
        prob_2 = 0
        
        trigramL = w_2 + ' ' + w_1 + ' ' + word
        prob_2 = getProbability(trigramL,trigrams,totalTris)
        
        prob_1 = 0
        bigramL = w_1 + ' ' + word
        prob_1 = getProbability(bigramL,bigrams,totalBis)
        
        prob = getProbability(word,unigrams,totalUnis)
        
        prob1 = 0
        bigramR = word + ' ' + w1
        prob1 = getProbability(bigramR,bigrams,totalBis)
        

        prob2 = 0
        trigramR = word + ' ' + w1 + ' ' + w2
        prob2 = getProbability(trigramR,trigrams,totalTris)

        vector_fet = np.arange(numFeatures)
        vector_fet[0] = len_word
        vector_fet[1] = num_syl
        vector_fet[2] = len_sen
        vector_fet[3] = prob_2
        vector_fet[4] = prob_1
        vector_fet[5] = prob
        vector_fet[6] = prob1
        vector_fet[7] = prob2
        vector_fet[8] = class_word

        matrix[indexRow]=vector_fet
        indexRow += 1
        if indexRow == 10:
            break

    return matrix

path='./data/Wikipedia_Train1.tsv'
matrix_train=getMatrix(path)
print(matrix_train)

path='./data/Wikipedia_Dev1.tsv'
#load_dataset(path)
matrix_dev=getMatrix(path)
print(matrix_dev)

numCol=matrix_train.shape[1]
X_train=matrix_train[:,0:numCol-1]
print(X_train)
y_train=matrix_train[:, -1] #last column
print(y_train)

numCol=matrix_dev.shape[1]
X_dev=matrix_dev[:,0:numCol-1]
print(X_dev)
y_dev=matrix_dev[:, -1] #last column
print(y_dev)

#Array with possible algorithms
classifiers = [
    svm.SVC()
]

for item in classifiers:
    print(item)
    clf = item
    #Train
    clf.fit(X_train, y_train)
    #Predict
    predicted=clf.predict(X_dev)
    #Precision, recall and f1 comparing gold standard (y_dev) with predictions
    bPrecis, bRecall, bFscore, bSupport = pr(y_dev, predicted, average='binary')
    #mostramos resultados
    print(bPrecis, bRecall, bFscore, bSupport)