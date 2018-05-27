# -*- coding: utf-8 -*-

import os

import scrapy
from scrapy.crawler import CrawlerProcess

class MashableSpider(scrapy.Spider):
    
    class TextItem(scrapy.Item):
        title = scrapy.Field()
        img_url = scrapy.Field()
        text = scrapy.Field()
        pass

    def __init__(self, url):
        self.start_urls = [url]

    def parse(self, response):
        article_item = self.TextItem()
        bunch_articles = []
        title = response.xpath('//div[@id="main"]//header[@class="article-header"]//h1[@class="title"]/text()').extract_first()
        img = response.xpath('//div[@id="main"]//header[@class="article-header"]//figure[@class="article-image"]//img[@class="microcontent"]/@src').extract_first()
        info = response.xpath('//div[@id="main"]//section[@class="article-content blueprint"]//p[not((parent::blockquote) or @class="see-also-link") and not(parent::div[@class="image-credit"])]')
 
        article_item['title'] = title
        article_item['img_url'] = img

        text = ""
        for p in info:
            line = "".join(p.xpath('.//text()').extract())
            text += " " + line
        article_item["text"] = text
        yield article_item
        

class MashableCrawler(object):
    
    def __init__(self):
        
        # Store module location
        self.package_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Path to files
        self.article_path = os.path.join(self.package_directory, 'article.json')
        
        return
    
    def retrieve(self, url):
        with open(self.article_path, 'w') as article:
            article.truncate(0)
        process = CrawlerProcess({'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)', 'FEED_URI': 'file:' + self.article_path})
        process.crawl(MashableSpider, url=url)
        process.start()