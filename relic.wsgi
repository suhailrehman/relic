python_home = '/var/www/relic/venv'

activate_this = python_home + '/bin/activate_this.py'
exec(open(activate_this).read(), dict(__file__=activate_this))

import sys
sys.path.insert(0,"/var/www/relic/")


import os
from dotenv import load_dotenv

load_dotenv('/var/www/relic/.env')


from relic.server import app as application
application.secret_key = os.getenv('SECRET_KEY', 'dev')
