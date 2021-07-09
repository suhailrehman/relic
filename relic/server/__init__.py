from relic.server.app import app

# Prep canned demos
CANNED_DEMO_LOCATION = "/home/suhail/Scratch/canned.tgz"
RELIC_TMP_LOCATION = "/tmp/relic/"

import tarfile
import os
os.makedirs(RELIC_TMP_LOCATION, exist_ok=True)
tar = tarfile.open(CANNED_DEMO_LOCATION)
tar.extractall(path="/tmp/relic/")
tar.close()


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
