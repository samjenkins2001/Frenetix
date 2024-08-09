document.getElementById('simulation-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const samplingDepth = document.getElementById('sampling_depth').value;
    const numTrajectories = document.getElementById('num_trajectories').value;

    const data = {
        sampling_depth: samplingDepth,
        num_trajectories: numTrajectories
    };

    fetch('/update_config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.textContent = `Output: ${data.output} Error: ${data.error}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
