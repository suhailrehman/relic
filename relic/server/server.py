import logging

import networkx as nx
from flask import Flask, render_template, request, redirect, url_for, send_file
from datetime import datetime
import os

import relic.graphs.graphs
from relic.core import main

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/')
@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    result_graph_file = f'/tmp/relic/{job_id}/inferred_graph.csv'
    job_log_file = f'/tmp/relic/{job_id}/job_status.log'
    logging.debug(f'Graph File: {result_graph_file}, {os.path.exists(result_graph_file)}')
    logging.debug(f'Job File: {job_log_file}')
    if os.path.exists(result_graph_file):
        logging.debug(f'Job completed, sending graph file')
        return redirect(url_for('render', job_id=job_id))
    else:
        return send_file(job_log_file, mimetype='text/plain')


@app.route('/render/<job_id>', methods=['GET'])
def render(job_id):
    result_graph_file = f'/tmp/relic/{job_id}/inferred_graph.csv'
    inferred_dir = f'/tmp/relic/{job_id}/inferred/'
    artifact_dir = f'/tmp/relic/{job_id}/artifacts/'
    rendered_graph = f'/tmp/relic/{job_id}/g_inferred.html'
    logging.debug(f'Graph File: {result_graph_file}, {os.path.exists(result_graph_file)}')
    if os.path.exists(rendered_graph):
        return send_file(rendered_graph, mimetype='text/html')
    elif os.path.exists(result_graph_file):
        logging.debug(f'Job completed, rendering graph')
        g_inferred = nx.read_edgelist(result_graph_file)
        network = relic.graphs.graphs.draw_web_graph(g_inferred, artifact_dir, inferred_dir)
        network.show(rendered_graph)
        return send_file(rendered_graph, mimetype='text/html')
    else:
        return "RELIC is still processing your job"


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        upload_fullpath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        f.save(upload_fullpath)
        job_id = datetime.now().strftime("%Y%m%d%H%M%S")
        output_folder = f'/tmp/relic/{job_id}/'

        args = ['relic', f"--artifact_zip={upload_fullpath}",
                f"--nb_name={job_id}",
                f"--out={str(output_folder)}"]

        main(args=args[1:])
        return redirect(url_for('status', job_id=job_id))


app.run(host='0.0.0.0', port=8000)