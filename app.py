from flask import Flask, request, jsonify, send_from_directory
import yaml
import sys
import logging
import subprocess

app = Flask(__name__)

last_input = {}

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

def update_planning_yaml(config_file, config_updates):
    """Update the planning.yaml file with new values."""
    config = load_yaml(config_file)
    config.update(config_updates)
    save_yaml(config, config_file)

@app.route('/update_config', methods=['POST'])
def update_config():
    global last_input
    data = request.json

    # Extract and validate sampling_depth
    sampling_depth = data.get('sampling_depth')
    if sampling_depth is None or not isinstance(sampling_depth, int) or sampling_depth < 1:
        return jsonify({'error': 'Invalid sampling_depth provided.'}), 400

    # Initialize config updates
    config_updates = {'sampling_depth': sampling_depth}

    # Process spacing_traj
    spacing_traj = data.get('spacing_traj', [])
    num_trajectories = data.get('num_trajectories', [])
    trajectory_spacing = data.get('trajectory_spacing', [])

    print("spacing_traj:", spacing_traj)
    print("num_trajectories:", num_trajectories)
    print("trajectory_spacing:", trajectory_spacing)


    if len(spacing_traj) != 2:
        return jsonify({'error': 'Invalid spacing_traj data.'}), 400
    config_updates['spacing_trajs'] = spacing_traj

    enable_spacing, enable_trajectories = spacing_traj

    if enable_trajectories:
        if len(num_trajectories) != sampling_depth:
            return jsonify({'error': 'Number of trajectories data does not match sampling depth.'}), 400
        config_updates['trajectories'] = num_trajectories

    if enable_spacing:
        if len(trajectory_spacing) != sampling_depth:
            return jsonify({'error': 'Trajectory spacing data does not match sampling depth.'}), 400
        config_updates['spacing'] = trajectory_spacing

    # Check if input has changed to avoid redundant processing
    if data != last_input:
        config_file = 'configurations/frenetix_motion_planner/planning.yaml'
        update_planning_yaml(config_file, config_updates)

        # Run the simulation script
        result = subprocess.run(['python3', 'main.py'], stdout=sys.stdout, stderr=sys.stderr, text=True)

        return jsonify({'output': result.stdout, 'error': result.stderr})

    else:
        return jsonify({'message': 'No new input provided. Simulation not run.'})

if __name__ == '__main__':
    app.run(debug=True)
