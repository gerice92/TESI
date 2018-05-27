from bottle import route, request, run, template, os
import subprocess
from simplify_web import get_html_article

@route('/hello')
def index():
    print("Got request...")
    url = request.query.url
    output = simplify(url)
    return template(output)

run(host='localhost', port=8080)