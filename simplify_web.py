#!/usr/bin/env python3

from classifier.word_classifier import WordClassifier
from synonyms.synonym_replace import SynonymReplace
from website.web_generator import WebGenerator
from website.web_launcher import WebLauncher


def main():
    wc = WordClassifier()
    #wc.train_classifier('svc')
    output = wc.classify("this is a test of the complex word classifier, does it work strange thing huh?")
    print(output)
    return

if __name__ == "__main__":
    main()