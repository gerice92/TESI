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
import sys

from multiprocessing import Process, Queue
from twisted.internet import reactor
from scrapy.crawler import CrawlerRunner

def main(start_url):
    global runner

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
    #CrawlingProcess.crawl(Article_crawler, start_url = start_url)
    #CrawlingProcess.start()
    
    run_spider(start_url)

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
        new_sentences.append("<p>" + sr_result + "</p>")

    title = data["title"]
    img = data["img_url"]
    text = " ".join(new_sentences)
    article = [title,img,text]
    
    return article

def get_html_article(start_url):
    article = main(start_url)
    html_article = WebGenerator.webGenerator(article[0],article[1],article[2])
    return html_article

def run_spider(start_url):
    def f(q):
        try:
            runner = CrawlerRunner({'FEED_URI': 'file:article.json',})
            deferred = runner.crawl(Article_crawler, start_url = start_url)
            deferred.addBoth(lambda _: reactor.stop())
            reactor.run()
            q.put(None)
        except Exception as e:
            q.put(e)

    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    result = q.get()
    p.join()

    if result is not None:
        raise result

if __name__ == "__main__":
   article = main("https://mashable.com/2018/04/28/facebook-fake-news-smaller-feed/#JMBs_1D1Paqs")
   #print(article[0])
   #print(article[1])
   #print(article[2])
   #WebGenerator.webGenerator(article[0],article[1],article[2])
   #print(sys.argv[1])
   #print("------")
   article = main(sys.argv[1])
   #print(article[0])
   #print(article[1])
   #print(article[2])
   html_article = WebGenerator.webGenerator(article[0],article[1],article[2])
   print(html_article)
