import logging
from threading import Thread

import networkx as nx
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, make_response
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from datetime import datetime
import os
import sys

import relic.graphs.graphs
from relic.core import main
from bs4 import BeautifulSoup



UPLOAD_FOLDER = '/tmp/uploads/'

app = Flask(__name__)
auth = HTTPBasicAuth()

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.debug = True

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

log = logging.getLogger('relic.graphs.graphs')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
log.addHandler(handler)
log.setLevel(logging.DEBUG)

log2 = logging.getLogger('relic.utils.serialize')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
log2.addHandler(handler)
log2.setLevel(logging.DEBUG)


try:
    USERNAME = os.environ['FLASK_USER']
    PASSWORD = generate_password_hash(os.environ['FLASK_PASSWORD'])
except KeyError as e:
    app.logger.fatal('You must set FLASK_USER and FLASK_PASSWORD environment variables to start this server')
    sys.exit(0)


@auth.verify_password
def verify_password(username, password):
    if username == USERNAME and \
            check_password_hash(PASSWORD, password):
        return username


@app.route('/')
@auth.login_required
def index():
    return render_template('index.html')


@app.route('/status/<job_id>', methods=['GET'])
@auth.login_required
def status(job_id):
    job_status_file = f'/tmp/relic/{job_id}/job_status.json'
    #app.logger.debug(f'Job Status File: {job_status_file}')
    if os.path.exists(job_status_file):
        return send_file(job_status_file, mimetype='text/json')
    else:
        return jsonify({'job_id': job_id, 'status': 'pending'})


@app.route('/log/<job_id>', methods=['GET'])
@auth.login_required
def log(job_id):
    job_log_file = f'/tmp/relic/{job_id}/job.log'
    app.logger.debug(f'Job File: {job_log_file}')
    if os.path.exists(job_log_file):
        return send_file(job_log_file, mimetype='text/plain')


@app.route('/render/<job_id>', methods=['GET'])
@auth.login_required
def render(job_id):
    width = float(request.args.get('width', default=5000))
    height = float(request.args.get('height', default=5000))
    result_graph_file = f'/tmp/relic/{job_id}/inferred_graph.csv'
    inferred_dir = f'/tmp/relic/{job_id}/inferred/'
    artifact_dir = f'/tmp/relic/{job_id}/artifacts/'
    rendered_graph = f'/tmp/relic/{job_id}/g_inferred.html'
    g_truth_file = f'/tmp/relic/{job_id}/true_graph.txt'
    app.logger.debug(f'Graph File: {result_graph_file}, {os.path.exists(result_graph_file)}')
    app.logger.debug(f'Ground Truth File: {g_truth_file}, {os.path.exists(g_truth_file)}')
    # if os.path.exists(rendered_graph):
    #    return send_file(rendered_graph, mimetype='text/html')
    if os.path.exists(result_graph_file):
        app.logger.debug(f'Job completed, rendering graph')
        g_inferred = nx.read_edgelist(result_graph_file)
        if os.path.exists(g_truth_file):
            g_truth = nx.read_edgelist(g_truth_file, create_using=nx.DiGraph)
        else:
            g_truth = None
        network = relic.graphs.graphs.draw_web_graph(g_inferred, artifact_dir, inferred_dir,
                                                     g_truth=g_truth, height=height, width=width)
        network.save_graph(rendered_graph)
        return send_file(rendered_graph, mimetype='text/html')
    else:
        return "RELIC is still processing your job"


@app.route('/artifact/<job_id>/<artifact_id>', methods=['GET'])
@auth.login_required
def artifact(job_id, artifact_id):
    artifact_file = f'/tmp/relic/{job_id}/artifacts/{artifact_id}'
    if os.path.exists(artifact_file):
        pd.set_option('display.max_colwidth', 20)
        df = pd.read_csv(artifact_file, index_col=0).head(100)
        html = df.to_html(notebook=True, classes="table table-striped table-sm", table_id="artifacttable", justify="unset")
        soup = BeautifulSoup(html, features="html.parser")
        soup.style.decompose()
        return make_response(str(soup))
    else:
        return make_response(jsonify({'job_id': job_id, 'artifact_id': artifact_id, 'status': 404}), 404)


@app.route('/export/<job_id>/', methods=['GET'])
@auth.login_required
def export(job_id):
    result_graph_file = f'/tmp/relic/{job_id}/inferred_graph.csv'
    if os.path.exists(result_graph_file):
        return send_file(result_graph_file, mimetype='text/csv', as_attachment=True, download_name=str(job_id)+'.csv')
    else:
        return make_response(jsonify({'job_id': job_id, 'status': 404}), 404)


@app.route('/upload', methods=['POST'])
@auth.login_required
def upload():
    f = request.files['artifacts']
    logger.info(request.form)
    upload_fullpath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    f.save(upload_fullpath)

    job_id = datetime.now().strftime("%Y%m%d%H%M%S")
    output_folder = f'/tmp/relic/{job_id}/'
    latest_symlink = f'/tmp/relic/latest/'

    if not os.path.isdir(os.path.dirname(output_folder)):
        os.makedirs(os.path.dirname(output_folder), exist_ok=True)

    # Todo: Debug inability to create symlink under WSL
    # os.symlink(output_folder, latest_symlink)

    args = ['relic', f"--artifact_zip={upload_fullpath}",
            f"--nb_name={job_id}",
            f"--out={str(output_folder)}"]

    args.extend(["--"+str(k)+"="+v[0] for k, v in request.form.to_dict(flat=False).items()])

    if 'g_truth' in request.files and request.files['g_truth'].filename:
        gt = request.files['g_truth']
        g_truth_full_path = os.path.join(app.config['UPLOAD_FOLDER'], gt.filename)
        gt.save(g_truth_full_path)
        args.append(f'--g_truth_file={g_truth_full_path}')

    thread = Thread(target=main, kwargs={'args': args[1:]})
    thread.daemon = True
    thread.start()
    return jsonify({'job_id': str(job_id),
                    'status': 'pending'})


app.run(host='0.0.0.0', port=8000)
