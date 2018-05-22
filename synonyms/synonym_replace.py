# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import wordnet as wn

class SynonymReplace(object):

    def __init__(self):
        return

    def word_swap(self, phrase, vec):
        """Return the phrase with synonyms instead complex words
        
        Keyword arguments:
        phrase -- original phrase with complex words detected
        vec -- array with [[complex_word_1,start,end],...]
        """
        new_phrase = phrase

        for vec_word in reversed(vec):
            word = vec_word[0]
            start = vec_word[1]
            end = vec_word[2] - 1

            synonyms = set()
            for ss in wn.synsets(word):
                for lm in ss.lemmas():
                    synonyms.add(lm.name())
            if synonyms == {word}:
                synonym = word
            else:
                synonyms.remove(word.lower())
                synonym = synonyms.pop()
            new_phrase = new_phrase[:start] + synonym + new_phrase[end+1:]                
            
        return new_phrase
