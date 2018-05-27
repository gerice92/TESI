# -*- coding: utf-8 -*-
import codecs
import requests
import webbrowser
import os

class WebGenerator(object):
    
    def __init__(self):
        
        # Store module location
        self.package_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Path to files
        self.template_path = os.path.join(self.package_directory, 'clean_template.html')
        self.result_path = os.path.join(self.package_directory, 'accesible_web.html')
        
        return

    def generate(self, title, img, text):
        
        page = str()
        
        with codecs.open(self.template_path, 'r') as template:
            page = template.read()
            page = page.replace('article_img', img)
            page = page.replace('<title></title>', '<title>' + title + '</title>')
            page = page.replace("<h1 class='title'></h1>","<h1 class='title'>" + title + "</h1>")
            page = page.replace('<p class="main_article"></p>','<p class="main_article">' + text + '<p>')

        with open(self.result_path, 'w') as result:
            result.write(page)

        return page
        
    def launch(self):
        webbrowser.open_new_tab('file://' + os.path.realpath(self.result_path))