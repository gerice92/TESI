# -*- coding: utf-8 -*-
import codecs
import requests
import webbrowser
import os

class WebGenerator(object):
    
    def __init__(self):
        return

    
    def webGenerator(title, img, text):
        f=codecs.open("website/clean_template.html", 'r')
        page = f.read()
        page_original = page

        page = page.replace('article_img',img)
        page = page.replace("<h1 class='title'></h1>","<h1 class='title'>" + title + "</h1>")
        page = page.replace('<p class="main_article"></p>','<p class="main_article">' + text + '<p>')

        html_file= open("accesible_web.html","w")
        html_file.write(page)
        html_file.close()
        
        # Open URL in a new tab, if a browser window is already open.
        webbrowser.open_new_tab('file://' + os.path.realpath("accesible_web.html"))
        #webbrowser.close()