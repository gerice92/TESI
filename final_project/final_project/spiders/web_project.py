import scrapy


class Article:
    title = ""
    extract = ""
    img_src = ""


class Articles_crawler(scrapy.Spider):
    name = "articles_crawler"

    def start_requests(self):
        urls = [
            'https://mashable.com/category/news-feed/',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        bunch_articles = []
        articles = response.xpath('//section//div[@class="story-stream"]//div[@class="article-container"]')
        for art in articles:
            article = Article()
            title = art.xpath('.//h2[@class="article-title"]/a/text()').extract_first()
            extract = art.xpath('.//div[@class="article-content"]//p/text()').extract_first()
            img = art.xpath('.//div[@class="article-img-container"]//img/@src').extract_first()
            article.title = title.strip()
            article.extract = extract.strip()
            article.img_src = img
            #print(article.title)
            #print(article.extract)
            #print(article.img_src)

            bunch_articles.append(article)
        print("--------")
        for bunch in bunch_articles:
            print(bunch.title)
            print(bunch.extract)
            print(bunch.img_src)
        print("--------")