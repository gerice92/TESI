#!/usr/bin/env python3

from classifier.word_classifier import WordClassifier
from synonyms.synonym_replace import SynonymReplace
from website.web_generator import WebGenerator
from website.web_launcher import WebLauncher
from crawler.crawler.spiders.web_article_project import Article_crawler
from scrapy.crawler import CrawlerProcess
import webbrowser
import unicodedata
import json
import os

def main(start_url):
 
    #Lauch Crawler
    CrawlingProcess = CrawlerProcess({
        'FEED_URI': 'file:article.json',
    })
    try:
        os.remove('article.json')
    except OSError:
        pass
    CrawlingProcess.crawl(Article_crawler, start_url = start_url)
    CrawlingProcess.start()

    #read json
    with open('article.json', encoding='utf-8', newline = "\n") as f:
        data = json.load(f)
        data_text = unicodedata.normalize("NFKD", data['text'])
        data_in_lines = [line.replace('.', '') for line in data_text.split(". ")]
    print(data_in_lines)

    #Init wordClassifier
    wc = WordClassifier()
    wc.train_classifier('naive-bayes')
    #wc.test_classifiers()
    new_sentences = list()
    #for line in ["this is a test of the complex word classifier, does it work strange thing huh?", "Evergreen trees are a symbol of fertility because they do not die in the winter"]:
    for line in data_in_lines:
        wc_result = wc.classify(line)
        sr = SynonymReplace()
        sr_result = sr.word_swap(line, wc_result)
        new_sentences.append(sr_result)
    print(new_sentences)

    title = data["title"]
    img = data["img_url"]
    text = " ".join(new_sentences)
    text = text.replace("\t"," ")
    article = [title,img,text]
    
    return article

if __name__ == "__main__":
   article = main("https://mashable.com/2018/04/28/facebook-fake-news-smaller-feed/#JMBs_1D1Paqs")
   print(article[0])
   print(article[1])
   print(article[2])
   WebGenerator.webGenerator(article[0],article[1],article[2])