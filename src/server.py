#!/usr/bin/env python3

import subprocess
import os

from bottle import route, request, run
from simplify_web import simplify

@route('/hello')
def index():
    scr_path = os.path.dirname(__file__)
    smp_path = os.path.join(scr_path, 'simplify_web.py')
    subprocess.call(['./simplify_web.py', request.query.url])

run(host='localhost', port=8080)