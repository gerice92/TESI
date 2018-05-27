#!/usr/bin/env python3

import argparse
import webbrowser
import unicodedata
import json

from nltk import sent_tokenize

from crawler.mashable_crawler import MashableCrawler
from classifier.word_classifier import WordClassifier
from synonyms.synonym_replace import SynonymReplace
from website.web_generator import WebGenerator

def simplify(url):
    """Return article segments from URL"""

    # Train/Test classifiers (Optional step)
    wc = WordClassifier()
    wc.test_classifiers()
    wc.train_classifier('naive-bayes')
    
    # Start crawling process
    cr = MashableCrawler()
    cr.retrieve(url)

    # Read crawler output
    with open(cr.article_path, encoding='utf-8', newline = "\n") as f:
        data = json.load(f)
        data_text = unicodedata.normalize("NFKD", data['text'])
        data_in_lines = [sent.strip().replace("\"", "\'") for sent in sent_tokenize(data_text)]

    # Operate over text sentences
    new_sentences = list()
    for line in data_in_lines:
        wc_result = wc.classify(line) # Classify words in line
        sr = SynonymReplace()
        sr_result = sr.word_swap(line, wc_result) # Replace complex with simple
        new_sentences.append("<p>" + sr_result + "</p>")

    title = data["title"]
    img = data["img_url"]
    text = " ".join(new_sentences)
    
    # Start accesible web generation process
    wg = WebGenerator()
    article = wg.generate(title, img, text)
    return article
    
def launch():
    
    # Start accesible web launch process
    wg = WebGenerator()
    wg.launch()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Simplify text from a Mashable.com web page.')
    parser.add_argument('url', type=str, help='URL of the web page')
    args = parser.parse_args()
    simplify(args.url)
    launch() 
