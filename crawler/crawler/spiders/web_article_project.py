# -*- coding: utf-8 -*-

import scrapy
import os
from scrapy.crawler import CrawlerProcess
from scrapy.conf import settings
from crawler.crawler.items import TextItem 

class Article_crawler(scrapy.Spider):
    name = "article_crawler"

    def __init__(self, *args, **kwargs): 
      super(Article_crawler, self).__init__(*args, **kwargs) 

      self.start_urls = [kwargs.get('start_url')] 

    '''
    def start_requests(self):
        urls = [
            'https://mashable.com/2016/09/02/facebook-what-friends-are-talking-about/#daq3gTXT4kq8',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    '''

    def parse(self, response):
        article_item = TextItem()
        bunch_articles = []
        title = response.xpath('//div[@id="main"]//header[@class="article-header"]//h1[@class="title"]/text()').extract_first()
        img = response.xpath('//div[@id="main"]//header[@class="article-header"]//figure[@class="article-image"]//img[@class="microcontent"]/@src').extract_first()
        info  = response.xpath('//div[@id="main"]//section[@class="article-content blueprint"]//p[not((parent::blockquote) or @class="see-also-link") and not(parent::div[@class="image-credit"])]')
 
        article_item['title'] = title
        article_item['img_url'] = img

        text = ""
        for p in info:
            line = "".join(p.xpath('.//text()').extract())
            text += line
        article_item["text"] = text
        yield article_item
         
        print("file created!")



