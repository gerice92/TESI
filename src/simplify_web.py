#!/usr/bin/env python3

from classifier.word_classifier import WordClassifier
from synonyms.synonym_replace import SynonymReplace
from website.web_generator import WebGenerator
from website.web_launcher import WebLauncher
from crawler.crawler.spiders.web_article_project import Article_crawler
from scrapy.crawler import CrawlerProcess
from nltk import sent_tokenize
import webbrowser
import unicodedata
import json
import os

def main(start_url):
    
    # Train/Test classifiers (Optional step)
    wc = WordClassifier()
    wc.train_classifier('naive-bayes')
    wc.test_classifiers()
 
    # Run crawling process
    CrawlingProcess = CrawlerProcess({'FEED_URI': 'file:article.json',})
    try:
        os.remove('article.json')
    except OSError:
        pass
    CrawlingProcess.crawl(Article_crawler, start_url = start_url)
    CrawlingProcess.start()

    # Read crawler output
    with open('article.json', encoding='utf-8', newline = "\n") as f:
        data = json.load(f)
        data_text = unicodedata.normalize("NFKD", data['text'])
        data_in_lines = [sent.strip().replace("\"", "\'") for sent in sent_tokenize(data_text)]

    # Operate over text sentences
    new_sentences = list()
    for line in data_in_lines:
        wc_result = wc.classify(line) # Classify words in line
        sr = SynonymReplace()
        sr_result = sr.word_swap(line, wc_result) # Replace complex with simple
        new_sentences.append(sr_result)

    title = data["title"]
    img = data["img_url"]
    text = " ".join(new_sentences)
    article = [title,img,text]
    
    return article

if __name__ == "__main__":
   article = main("https://mashable.com/2018/04/28/facebook-fake-news-smaller-feed/#JMBs_1D1Paqs")
   print(article[0])
   print(article[1])
   print(article[2])
   WebGenerator.webGenerator(article[0],article[1],article[2])