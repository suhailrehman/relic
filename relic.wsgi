#!/usr/bin/python3

import sys
sys.path.insert(0, '/var/www/relic/')

import os
from dotenv import load_dotenv

load_dotenv('.env')


from relic.server import app as application
application.secret_key = os.getenv('SECRET_KEY', 'dev')