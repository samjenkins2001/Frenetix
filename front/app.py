from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.json
    parameter = data.get('parameter')
    value = data.get('value')

    try:
        # Run the main.py script with the provided parameter and value
        result = subprocess.run(['python', '../main.py', parameter, str(value)], capture_output=True, text=True)
        output = result.stdout
    except Exception as e:
        output = str(e)

    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True)
