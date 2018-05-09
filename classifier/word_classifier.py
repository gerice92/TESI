# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import word_tokenize
import csv
import string
import numpy as np
import os

class WordClassifier(object):
    
    NUM_FEATURES = 10
    ALGORITHM_SVM = 'svc'
    ALGORITHM_RDM_FST = 'random-forest'
    ALGORITHM_N_B = 'naive-bayes'
    
    def __init__(self):
        
        # Store module location
        self.package_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Path to files
        self.wiki_1gram_path = os.path.join(self.package_directory, 'models', 'wiki_1.wngram')
        self.wiki_2gram_path = os.path.join(self.package_directory, 'models', 'wiki_2.wngram')
        self.wiki_3gram_path = os.path.join(self.package_directory, 'models', 'wiki_3.wngram')
        self.wiki_train_path = os.path.join(self.package_directory, 'data', 'wiki_train_1.tsv')
        self.wiki_eval_path = os.path.join(self.package_directory, 'data', 'wiki_eval_1.tsv')
        self.classifier_out_path = os.path.join(self.package_directory, 'classifier.pkl')
        
        # Load unigrams
        dic_path = self.wiki_1gram_path
        self.unigrams = self._load_dic(dic_path)
        self.total_unigrams = sum(iter(self.unigrams.values()))

        # Loag bigrams
        dic_path = self.wiki_2gram_path
        self.bigrams = self._load_dic(dic_path)
        self.total_bigrams = sum(iter(self.bigrams.values()))

        # Load trigrams
        dic_path = self.wiki_3gram_path
        self.trigrams = self._load_dic(dic_path)
        self.total_trigrams = sum(iter(self.trigrams.values()))

    def _count_syll(self, word):
        """Return number of syllables for a given word
        
        Keyword arguments:
        word -- the word to calculate the number of syllables for
        """
        
        syll = lambda w:len(''.join(c if c in "aeiouy" else ' ' for c in w.rstrip('e')).split())
        num_syl = syll(word)
        return num_syl
        
    def _count_lemmas(self, word):
        """Return number of lemmas for a given word
        
        Keyword arguments:
        word -- the word to calculate the number of synonyms for
        """
        
        lemmas = 0
        
        for synset in wordnet.synsets(word):
            lemmas += len(synset.lemmas())
                
        return lemmas

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
        matrix = np.empty(shape = [num_examples, WordClassifier.NUM_FEATURES])

        # Parse dataset
        with open(path, encoding="utf8", newline='\n') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')

            indexRow = 0
            for row in tsvin:
                ide = row[0] # Paragraph ID
                sentence = row[1] # Sentence
                start = int(row[2]) # Starting character index of word in sentence
                end = int(row[3]) # Ending character index of word in sentence
                word = row[4] # Word to classify
                classification = int(row[9]) # Clasification: 0 means simple, 1 means complex
                probability = int(float(row[10]) * 100) # Word probability
            
                len_word = end - start # Word length
                num_syl = self._count_syll(word) # Word number of syllables
                len_sen = len(sentence) # Sentence length
                w_2,w_1,w1,w2 = self._get_window(word, sentence, start) # Window

                # Left trigram probability
                prob_2 = 0
                trigramL = w_2 + ' ' + w_1 + ' ' + word
                prob_2 = self._get_probability(trigramL, self.trigrams, self.total_trigrams)
            
                # Left bigram probability
                prob_1 = 0
                bigramL = w_1 + ' ' + word
                prob_1 = self._get_probability(bigramL, self.bigrams, self.total_bigrams)
        
                # Unigram probability
                prob = self._get_probability(word, self.unigrams, self.total_unigrams)
        
                # Right bigram probability
                prob1 = 0
                bigramR = word + ' ' + w1
                prob1 = self._get_probability(bigramR, self.bigrams, self.total_bigrams)
        
                # Right trigram probability
                prob2 = 0
                trigramR = word + ' ' + w1 + ' ' + w2
                prob2 = self._get_probability(trigramR, self.trigrams, self.total_trigrams)
                
                num_lem = self._count_lemmas(word) # Word number of lemmas

                # Prepare feature matrix for current sample
                vector_fet = np.arange(WordClassifier.NUM_FEATURES)
                vector_fet[0] = len_word
                vector_fet[1] = num_syl
                vector_fet[2] = len_sen
                vector_fet[3] = prob_2
                vector_fet[4] = prob_1
                vector_fet[5] = prob
                vector_fet[6] = prob1
                vector_fet[7] = prob2
                vector_fet[8] = num_lem
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
        matrix_train = self._get_matrix(self.wiki_train_path).astype(int)

        # Evaluation set from Wikipedia.org samples
        matrix_dev = self._get_matrix(self.wiki_eval_path).astype(int)

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

    def train_classifier(self, classifier):
        
        # Training set from Wikipedia.org samples
        matrix_train = self._get_matrix(self.wiki_train_path).astype(int)
        
        # Training set
        num_col = matrix_train.shape[1]
        X_train = matrix_train[:,0:num_col-1] # Feature array (all columns except last)
        y_train = matrix_train[:, -1] # Assigned class (last column)
        
        if classifier == WordClassifier.ALGORITHM_SVM:
            clf = SVC()
            clf.fit(X_train, y_train)
            joblib.dump(clf, WordClassifier.CLASSIFIER_OUTPUT)
        elif classifier == WordClassifier.ALGORITHM_RDM_FST:
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            joblib.dump(clf, WordClassifier.CLASSIFIER_OUTPUT)
        elif classifier == WordClassifier.ALGORITHM_N_B:
            clf = GaussianNB()
            clf.fit(X_train, y_train)
            joblib.dump(clf, self.classifier_out_path)
    
    def classify(self, sentence):
        
        # Load classifier object
        clf = joblib.load(self.classifier_out_path)
        
        # Tokenize sentence
        tokens = word_tokenize(sentence)
        
        # Recreate sentence from tokenization
        sentence = ' '.join(tokens)
        
        pos = 0
        token_dic = list()
        
        # Remove stop words
        for token in tokens:
            if token not in stopwords.words('english') and token not in string.punctuation:
                token_dic.append((token, pos, pos + len(token)))
            pos += len(token) + 1
        
        # Prepare feature matrix
        matrix = np.empty(shape = [1, WordClassifier.NUM_FEATURES - 1])
        
        # Calculate sentence length
        len_sen = len(sentence)
        
        res = list()
        
        for word, start, end in token_dic:
            
            len_word = end - start
            num_syl = self._count_syll(word)
            num_lem = self._count_lemmas(word)
            w_2,w_1,w1,w2 = self._get_window(word, sentence, start) # Window

            # Left trigram probability
            prob_2 = 0
            trigramL = w_2 + ' ' + w_1 + ' ' + word
            prob_2 = self._get_probability(trigramL, self.trigrams, self.total_trigrams)
        
            # Left bigram probability
            prob_1 = 0
            bigramL = w_1 + ' ' + word
            prob_1 = self._get_probability(bigramL, self.bigrams, self.total_bigrams)
    
            # Unigram probability
            prob = self._get_probability(word, self.unigrams, self.total_unigrams)
    
            # Right bigram probability
            prob1 = 0
            bigramR = word + ' ' + w1
            prob1 = self._get_probability(bigramR, self.bigrams, self.total_bigrams)
    
            # Right trigram probability
            prob2 = 0
            trigramR = word + ' ' + w1 + ' ' + w2
            prob2 = self._get_probability(trigramR, self.trigrams, self.total_trigrams)
            
            # Prepare feature matrix for current sample
            vector_fet = np.arange(WordClassifier.NUM_FEATURES - 1)
            vector_fet[0] = len_word
            vector_fet[1] = num_syl
            vector_fet[2] = len_sen
            vector_fet[3] = prob_2
            vector_fet[4] = prob_1
            vector_fet[5] = prob
            vector_fet[6] = prob1
            vector_fet[7] = prob2
            vector_fet[8] = num_lem
            
            matrix[0] = vector_fet
            
            predicted = clf.predict(matrix)
            
            if predicted[0] == 1:
                res.append((word, start, end))
                
        return res