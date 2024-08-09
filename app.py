from flask import Flask, request, jsonify, send_from_directory
import yaml
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

def load_yaml(file_path):
    """Load YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    """Save YAML file."""
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

def update_planning_yaml(config_file, sampling_depth, num_trajectories):
    """Update the planning.yaml file with new values."""
    config = load_yaml(config_file)
    config['sampling_depth'] = sampling_depth
    config['trajectories'] = num_trajectories
    save_yaml(config, config_file)

@app.route('/update_config', methods=['POST'])
def update_config():
    # Parse JSON input from the request
    data = request.json

    # Extract sampling_depth and num_trajectories from the input
    sampling_depth = data.get('sampling_depth')
    num_trajectories = data.get('num_trajectories')

    # Validate input
    if sampling_depth is None or num_trajectories is None:
        return jsonify({'error': 'Missing sampling_depth or num_trajectories'}), 400

    # Convert num_trajectories from a string to a list of floats
    try:
        sampling_depth = int(sampling_depth)
        num_trajectories = list(map(float, num_trajectories.split(',')))
    except ValueError:
        return jsonify({'error': 'Invalid format for num_trajectories'}), 400

    # Define the path to your YAML configuration file
    config_file = 'configurations/frenetix_motion_planner/planning.yaml'

    # Update the YAML configuration file
    update_planning_yaml(config_file, sampling_depth, num_trajectories)

    # Optionally, run another script (e.g., main.py) if needed
    result = subprocess.run(['python3', 'main.py'], capture_output=True, text=True)

    return jsonify({'output': result.stdout, 'error': result.stderr})

if __name__ == '__main__':
    app.run(debug=True)
