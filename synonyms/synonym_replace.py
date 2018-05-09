# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import wordnet as wn

class SynonymReplace(object):

    def __init__(self):
        return

    def word_swap(phrase,vec):
        """Return the phrase with synonyms instead complex words
        
        Keyword arguments:
        phrase -- original phrase with complex words detected
        vec -- array with [[complex_word_1,start,end],...]
        """
        new_phrase = phrase

        for vec_word in reversed(vec):
            word = vec_word[0]
            start = vec_word[1]
            end = vec_word[2]

            i=0
            for i, ss in enumerate(wn.synsets(word)):
                synonym = ss.name()
                print(synonym)
                if "." in synonym:
                    synonym = synonym.split(".")[0]
                #Need to define criteria for synonym selection
                if i == 2:
                    break

            if i==0:
                print('\t'+word + ' is not found in WordNet')
            else:
                new_phrase = new_phrase[:start] + synonym + new_phrase[end+1:]                
            
        return new_phrase
