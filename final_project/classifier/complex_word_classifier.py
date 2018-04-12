from __future__ import division
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support as pr
import csv
import re
import numpy as np

class WordClassifier(object):
    
    def __init__(self):
        self.NUM_FEATURES = 9
        self.WIKIPEDIA_1GRAM_PATH = 'models/wiki_1.wngram'
        self.WIKIPEDIA_2GRAM_PATH = 'models/wiki_2.wngram'
        self.WIKIPEDIA_3GRAM_PATH = 'models/wiki_3.wngram'
        self.WIKIPEDIA_TRAIN_SET = 'data/wiki_train_1.tsv'
        self.WIKIPEDIA_EVAL_SET = 'data/wiki_eval_1.tsv'

    def _count_syll(self, word):
        """Return number of syllables for a given word
        
        Keyword arguments:
        word -- the word to calculate the number of syllables for
        """
        
        syll = lambda w:len(''.join(c if c in "aeiouy" else ' ' for c in w.rstrip('e')).split())
        num_syl = syll(word)
        return num_syl

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
        
        # Return -1 if no occurrences found
        if num_ocurr == 0:
            raise ValueError("word has no ocurrences in sentence")
            
        # Return -1 if start is outside sentence range
        if start < 0 or start >= len(sentence):
            raise ValueError("start is out of range of the sentence")

        # Get word index in the sentence
        index = -1
        if num_ocurr == 1 or start == 0:
            index = tokens.index(word)
        else:
            index = len(sentence[0:start-1].split(' ')) 
            if index >= 0 and tokens[index] != word:
                raise ValueError("word not found at start position")

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
        f = open(path, 'r')
        for line in f:
            line = line.strip()
            pos = line.rfind(' ')
            key = line[0:pos]
            freq = line[pos+1:len(line)]
            dic[key] = int(freq)
        f.close()
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
        
        f = open(path, 'r') 
        num_examples = sum(1 for line in f)
        f.close()
        return num_examples

    def _get_matrix(self, path):
        """Return feature matrix from supervised examples"""
    
        num_examples = self._get_lines(path)
        num_features = self.NUM_FEATURES
        matrix = np.empty(shape = [num_examples, num_features])

        # Load unigrams
        dic_path = self.WIKIPEDIA_1GRAM_PATH
        unigrams = self._load_dic(dic_path)
        total_unigrams = sum(unigrams.itervalues())

        # Loag bigrams
        dic_path = self.WIKIPEDIA_2GRAM_PATH
        bigrams = self._load_dic(dic_path)
        total_bigrams = sum(bigrams.itervalues())

        # Load trigrams
        dic_path = self.WIKIPEDIA_3GRAM_PATH
        trigrams = self._load_dic(dic_path)
        total_trigrams = sum(trigrams.itervalues())

        # Parse dataset
        tsvin = open(path, "rb")
        tsvin = csv.reader(tsvin, delimiter="\t")

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

            # Prepare feature matrix for current example
            vector_fet = np.arange(num_features)
            vector_fet[0] = len_word
            vector_fet[1] = num_syl
            vector_fet[2] = len_sen
            vector_fet[3] = prob_2
            vector_fet[4] = prob_1
            vector_fet[5] = prob
            vector_fet[6] = prob1
            vector_fet[7] = prob2
            vector_fet[8] = classification

            # Store in global matrix
            matrix[indexRow] = vector_fet
        
            # Continue until the matrix is filled
            indexRow += 1
            if indexRow == num_examples:
                break
        
        return matrix
        
    def test_classifier(self):
        """Print statistics for multiple classifiers"""

        path = self.WIKIPEDIA_TRAIN_SET
        matrix_train = self._get_matrix(path)
        matrix_train = matrix_train.astype(int)

        path = self.WIKIPEDIA_EVAL_SET
        matrix_dev = self._get_matrix(path)
        matrix_dev = matrix_dev.astype(int)

        # Training set
        num_col = matrix_train.shape[1]
        X_train = matrix_train[:,0:num_col-1] # Feature array (all columns except last)
        y_train = matrix_train[:, -1] # Assigned class (last column)

        # Evaluation set
        num_col = matrix_dev.shape[1]
        X_dev = matrix_dev[:,0:num_col-1] # Feature array (all columns except last)
        y_dev = matrix_dev[:, -1] # Assigned class (last column)

        # Classification algorithms
        classifiers = [svm.SVC()]

        # Compare classifiers
        for item in classifiers:
            print(item)
            clf = item
            
            # Train with training set
            clf.fit(X_train, y_train)
            
            # Predict for evaluation set
            predicted = clf.predict(X_dev)
            
            # Precision, recall and f1 comparing expected classes with predicted classes
            bPrecis, bRecall, bFscore, bSupport = pr(y_dev, predicted, average='binary')
            
            # Print statistics
            print(bPrecis, bRecall, bFscore, bSupport)
