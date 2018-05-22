#!/usr/bin/env python3

from classifier.word_classifier import WordClassifier
from synonyms.synonym_replace import SynonymReplace
from website.web_generator import WebGenerator
from website.web_launcher import WebLauncher

def main():
    wc = WordClassifier()
    wc.train_classifier('naive-bayes')
    #wc.test_classifiers()
    new_sentences = list()
    for line in ["this is a test of the complex word classifier, does it work strange thing huh?", "Evergreen trees are a symbol of fertility because they do not die in the winter"]:
        wc_result = wc.classify(line)
        sr = SynonymReplace()
        sr_result = sr.word_swap(line, wc_result)
        new_sentences.append(sr_result)
    print(new_sentences)
    return

if __name__ == "__main__":
    main()