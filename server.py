import os
import geopandas as gpd
from shapely.geometry import Polygon
from flask import Flask, request, send_from_directory, safe_join
from flask_cors import CORS
import numpy as np
import json

app = Flask(__name__, static_folder=os.path.abspath('../urban-behaviour/src/app/'))
cors = CORS(app)
geo_network = None
gdf_network = None

@app.route('/', methods=['GET'])
def index():
    print("Trying to open index page")
    return serve_static('index.html')
#     return send_from_directory(safe_join(app.root_path, 'vis/src/'), 'index.html')

@app.route('/<path:filename>', methods=['GET'])
def serve_static(filename):
    return send_from_directory(safe_join(app.root_path,'urban-behaviour/dist/urban-behaviour'), filename)

@app.route('/network', methods=['GET'])
def serve_network():
    # print("Sending requested street shadow data", geo_network)
    return geo_network

@app.route('/distribution', methods=['POST'])
def serve_distribution():
    # print("Incoming distribution request: ", request.data)
    data = json.loads(request.data)

    if len(data) == 0:
        return 

    summary_stats = {'stats': [compute_distribution(data[key], key) for key in data.keys()]}
    print(summary_stats)

    return summary_stats

def compute_distribution(shadow_values, key):

    values = np.array(shadow_values)

    if values.size == 0:
        return {'key': key, 'min': 0, 'q1': 0, 'median': 0, 'q3': 0, 'max': 0, 'interQuantileRange': 0}

    mini = values.min()
    maxi = values.max()
    q1 = np.quantile(values, q=0.25)
    median = np.quantile(values, q=0.5)
    q3 = np.quantile(values, q=0.75)
    iqr = q3-q1

    stats = {'key': key, 'min': mini, 'q1': q1, 'median': median, 'q3': q3, 'max': maxi, 'interQuantileRange': iqr}
    
    return stats

def load():
    global gdf_network
    global geo_network
    gdf_network = gpd.read_file('./chicago-street-shadow.geojson')
    geo_network = gdf_network.to_json()

if __name__ == '__main__':
    load()
    app.run(debug=True, host='127.0.0.1', port=8080)