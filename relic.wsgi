#!/usr/bin/python3

import sys
sys.path.insert(0, '/var/www/relic/')

from relic.server import app as application

#virtualenv
# activate_this = '/home/cogsci-cnddcollab/FlaskApp/venv/bin/activate_this.py'
# with open(activate_this) as file_:
#     exec(file_.read(), dict(__file__=activate_this))

# import sys
# import logging
# logging.basicConfig(stream=sys.stderr)
# sys.path.insert(0, '/var/www/FlaskApp')
#
# from FlaskApp import app as application