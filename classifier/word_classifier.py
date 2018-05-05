# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib
from nltk.corpus import wordnet
import csv
import numpy as np

class WordClassifier(object):
    
    def __init__(self):
        self.NUM_FEATURES = 10
        self.WIKIPEDIA_1GRAM_PATH = 'models/wiki_1.wngram'
        self.WIKIPEDIA_2GRAM_PATH = 'models/wiki_2.wngram'
        self.WIKIPEDIA_3GRAM_PATH = 'models/wiki_3.wngram'
        self.WIKIPEDIA_TRAIN_SET = 'data/wiki_train_1.tsv'
        self.WIKIPEDIA_EVAL_SET = 'data/wiki_eval_1.tsv'
        self.ALGORITHM_SVM = 'svc'
        self.CLASSIFIER_OUTPUT = 'classifier.pkl'

    def _count_syll(self, word):
        """Return number of syllables for a given word
        
        Keyword arguments:
        word -- the word to calculate the number of syllables for
        """
        
        syll = lambda w:len(''.join(c if c in "aeiouy" else ' ' for c in w.rstrip('e')).split())
        num_syl = syll(word)
        return num_syl
        
    def _count_syns(self, word):
        """Return number of synonyms for a given word
        
        Keyword arguments:
        word -- the word to calculate the number of synonyms for
        """
        
        synonyms = []
        
        for synonym in wordnet.synsets(word):
            for lemma in synonym.lemmas():
                synonyms.append(lemma.name())
                
        return len(synonyms)

    def _get_window(self, word, sentence, start):
        """Return words surrounding a given word in a sentence, or -1 if not present
        
        Keyword arguments:
        word -- the word to calculate the window for
        sentence -- the sentence containing the word
        start -- the character index for the word in the sentence
        """

        # Split sentence by whitespaces to get all tokens
        tokens = sentence.split()
        
        # Count tokens
        num_tokens = len(tokens)
        
        # Parse start as integer
        start = int(start)

        # Prepare two-word window
        w_2 = str()
        w_1 = str()
        w1 = str()
        w2 = str()

        # Count word appearances in the sentence
        num_ocurr = tokens.count(word)

        if num_ocurr == 0:
            raise ValueError("word has no ocurrences in sentence")

        if start < 0 or start >= len(sentence):
            raise ValueError("start is out of range of the sentence")

        # Get word index in the sentence
        index = -1
        if num_ocurr == 1 or start == 0:
            index = tokens.index(word)
        else:
            index = len(sentence[0:start-1].split())
            if tokens[index] != word:
                raise ValueError("word ({}) not found at start position ({})".format(word, start))

        # Prepare window values
        if index-1 >= 0:
            w_1 = tokens[index-1] 
            if index-2 >= 0:
                w_2 = tokens[index-2]
        if index+1 <= num_tokens-1:
            w1 = tokens[index+1]
            if index+2 <= num_tokens-1:
                w2 = tokens[index+2]
                            
        return w_2,w_1,w1,w2

    def _load_dic(self, path):
        """Return dictionary with n-grams frequencies from file"""
        
        dic = dict()
        with open(path, encoding='utf8', newline='\n') as f:
            for line in f:
                line = line.strip()
                pos = line.rfind(' ')
                key = line[0:pos]
                freq = line[pos+1:len(line)]
                dic[key] = int(freq)
                
        return dic

    def _get_probability(self, ngram, dic, size):
        """Return n-gram probability relative to total count of n-grams
        
        Keyword arguments:
        ngram -- the gram to evaluate
        dic -- the n-gram frequency dictionary
        size -- the total number of n-grams
        """

        ngram = ngram.lower()
        ngram = ngram.strip()
        prob = 0
        try:
            prob = dic[ngram]
            prob = prob/size
        except:
            pass

        return prob

    def _get_lines(self, path):
        """Return number of lines in training set"""
        
        with open(path, encoding='utf8', newline='\n') as f:
            num_examples = sum(1 for line in f)
            
        return num_examples

    def _get_matrix(self, path):
        """Return feature matrix from supervised examples"""
    
        num_examples = self._get_lines(path)
        num_features = self.NUM_FEATURES
        matrix = np.empty(shape = [num_examples, num_features])

        # Load unigrams
        dic_path = self.WIKIPEDIA_1GRAM_PATH
        unigrams = self._load_dic(dic_path)
        total_unigrams = sum(iter(unigrams.values()))

        # Loag bigrams
        dic_path = self.WIKIPEDIA_2GRAM_PATH
        bigrams = self._load_dic(dic_path)
        total_bigrams = sum(iter(bigrams.values()))

        # Load trigrams
        dic_path = self.WIKIPEDIA_3GRAM_PATH
        trigrams = self._load_dic(dic_path)
        total_trigrams = sum(iter(trigrams.values()))

        # Parse dataset
        with open(path, encoding="utf8", newline='\n') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')

            indexRow = 0
            for row in tsvin:
                ide = row[0] # Paragraph ID
                sentence = row[1] # Sentence
                start = row[2] # Starting character index of word in sentence
                end = row[3] # Ending character index of word in sentence
                word = row[4] # Word to classify
                classification = row[9] # Clasification: 0 means simple, 1 means complex
                probability = row[10] # Word probability
            
                len_word = len(word) # Word length
                num_syl = self._count_syll(word) # Word number of syllables
                len_sen = len(sentence) # Sentence length
                w_2,w_1,w1,w2 = self._get_window(word, sentence, start) # Window

                # Left trigram probability
                prob_2 = 0
                trigramL = w_2 + ' ' + w_1 + ' ' + word
                prob_2 = self._get_probability(trigramL, trigrams, total_trigrams)
            
                # Left bigram probability
                prob_1 = 0
                bigramL = w_1 + ' ' + word
                prob_1 = self._get_probability(bigramL, bigrams, total_bigrams)
        
                # Unigram probability
                prob = self._get_probability(word, unigrams, total_unigrams)
        
                # Right bigram probability
                prob1 = 0
                bigramR = word + ' ' + w1
                prob1 = self._get_probability(bigramR, bigrams, total_bigrams)
        
                # Right trigram probability
                prob2 = 0
                trigramR = word + ' ' + w1 + ' ' + w2
                prob2 = self._get_probability(trigramR, trigrams, total_trigrams)
                
                num_syn = self._count_syns(word) # Word number of synonyms

                # Prepare feature matrix for current sample
                vector_fet = np.arange(num_features)
                vector_fet[0] = len_word
                vector_fet[1] = num_syl
                vector_fet[2] = len_sen
                vector_fet[3] = prob_2
                vector_fet[4] = prob_1
                vector_fet[5] = prob
                vector_fet[6] = prob1
                vector_fet[7] = prob2
                vector_fet[8] = num_syn
                vector_fet[9] = classification

                # Store in global matrix
                matrix[indexRow] = vector_fet
        
                # Continue until the matrix is filled
                indexRow += 1
                if indexRow == num_examples:
                    break
        
        return matrix
        
    def test_classifiers(self):
        """Print statistics for multiple classifiers"""

        # Training set from Wikipedia.org samples
        matrix_train = self._get_matrix(self.WIKIPEDIA_TRAIN_SET).astype(int)

        # Evaluation set from Wikipedia.org samples
        matrix_dev = self._get_matrix(self.WIKIPEDIA_EVAL_SET).astype(int)

        # Feature matrix for training set
        num_col = matrix_train.shape[1]
        X_train = matrix_train[:,0:num_col-1] # Feature array (all columns except last)
        y_train = matrix_train[:, -1] # Assigned class (last column)

        # Feature matrix for evaluation set
        num_col = matrix_dev.shape[1]
        X_dev = matrix_dev[:,0:num_col-1] # Feature array (all columns except last)
        y_dev = matrix_dev[:, -1] # Assigned class (last column)

        # Classification algorithms to test
        classifiers = [SVC(), RandomForestClassifier(), GaussianNB()]

        # Compare classifiers
        for item in classifiers:
            print(item)
            clf = item
            
            # Train with training set
            clf.fit(X_train, y_train)
            
            # Predict for evaluation set
            predicted = clf.predict(X_dev)
            
            # Precision, recall and f1 comparing expected classes with predicted classes
            bPrecis, bRecall, bFscore, bSupport = precision_recall_fscore_support(y_dev, predicted, average='binary')
            
            # Print statistics
            print(bPrecis, bRecall, bFscore, bSupport)

    def train_classifier(self):
        
        # Training set from Wikipedia.org samples
        matrix_train = self._get_matrix(self.WIKIPEDIA_TRAIN_SET).astype(int)
        
        # Training set
        num_col = matrix_train.shape[1]
        X_train = matrix_train[:,0:num_col-1] # Feature array (all columns except last)
        y_train = matrix_train[:, -1] # Assigned class (last column)
        
        clf = SVC()
        clf.fit(X_train, y_train)
        joblib.dump(clf, self.CLASSIFIER_OUTPUT)
    
    def classify(self, input):
        
        clf = joblib.load(self.CLASSIFIER_OUTPUT)
        # TODO: clf.predict() over input, assuming format is that of training set