from bottle import route, request, run, template, os
import subprocess
from simplify_web import get_html_article

@route('/hello')
def index():
    print("Got request...")
    url = request.query.url
    #output = subprocess.check_output("python3 simplify_web.py " + url, shell = True)
    output = get_html_article(url)
    #os.system("python3 simplify_web.py " + url)
    return template(output)
    # return template('<b>Hello {{url}}</b>!', url=url)

run(host='localhost', port=8080)