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
        open_div = '<span class="new_tooltip">'
        close_div = '</span>'

        for vec_word in reversed(vec):
            word = vec_word[0].lower()
            start = vec_word[1]
            end = vec_word[2] - 1
            tag_word = '<span class="new_tooltiptext">' + word +"</span>"

            synonyms = set()
            for ss in wn.synsets(word):
                for lm in ss.lemmas():
                    synonyms.add(lm.name())
            if len(synonyms) > 0 and synonyms != {word}:
                if word in synonyms:
                    synonyms.remove(word)
                synonym = min(synonyms, key=len)
                new_phrase = new_phrase[:start] + open_div + synonym + tag_word + close_div  + new_phrase[end+1:]    
        return new_phrase
