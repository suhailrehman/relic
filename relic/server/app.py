import logging
from threading import Thread

import networkx as nx
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, make_response
from datetime import datetime
import os

import relic.graphs.graphs
from relic.core import main

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.debug = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    job_status_file = f'/tmp/relic/{job_id}/job_status.json'
    logger.debug(f'Job Status File: {job_status_file}')
    if os.path.exists(job_status_file):
        return send_file(job_status_file, mimetype='text/json')
    else:
        return jsonify({'job_id': job_id, 'status': 'pending'})


@app.route('/log/<job_id>', methods=['GET'])
def log(job_id):
    job_log_file = f'/tmp/relic/{job_id}/job.log'
    logger.debug(f'Job File: {job_log_file}')
    if os.path.exists(job_log_file):
        return send_file(job_log_file, mimetype='text/plain')


@app.route('/render/<job_id>', methods=['GET'])
def render(job_id):
    result_graph_file = f'/tmp/relic/{job_id}/inferred_graph.csv'
    inferred_dir = f'/tmp/relic/{job_id}/inferred/'
    artifact_dir = f'/tmp/relic/{job_id}/artifacts/'
    rendered_graph = f'/tmp/relic/{job_id}/g_inferred.html'
    g_truth_file = f'/tmp/relic/{job_id}/true_graph.pkl'
    logger.debug(f'Graph File: {result_graph_file}, {os.path.exists(result_graph_file)}')
    logger.debug(f'Ground Truth File: {g_truth_file}, {os.path.exists(g_truth_file)}')
    if os.path.exists(rendered_graph):
        return send_file(rendered_graph, mimetype='text/html')
    elif os.path.exists(result_graph_file):
        logger.debug(f'Job completed, rendering graph')
        g_inferred = nx.read_edgelist(result_graph_file)
        g_truth = nx.read_gpickle(g_truth_file) if os.path.exists(g_truth_file) else None
        network = relic.graphs.graphs.draw_web_graph(g_inferred, artifact_dir, inferred_dir, g_truth=g_truth)
        network.save_graph(rendered_graph)
        return send_file(rendered_graph, mimetype='text/html')
    else:
        return "RELIC is still processing your job"


@app.route('/artifact/<job_id>/<artifact_id>', methods=['GET'])
def artifact(job_id, artifact_id):
    artifact_file = f'/tmp/relic/{job_id}/artifacts/{artifact_id}'
    if os.path.exists(artifact_file):
        pd.set_option('display.max_colwidth', 20)
        df = pd.read_csv(artifact_file, index_col=0).head(100)
        return df.to_html(notebook=True, classes="table table-striped table-sm", table_id="artifacttable")
    else:
        return make_response(jsonify({'job_id': job_id, 'artifact_id': artifact_id, 'status': 404}), 404)


@app.route('/upload', methods=['POST'])
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

    if 'g_truth' in request.files:
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
