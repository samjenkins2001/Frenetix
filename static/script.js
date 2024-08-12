document.getElementById('sampling_depth').addEventListener('input', function() {
    const depth = parseInt(this.value, 10);
    const container = document.getElementById('trajectories-container');
    container.innerHTML = ''; // Clear previous inputs

    if (depth > 0) {
        for (let i = 0; i < depth; i++) {
            const label = document.createElement('label');
            label.textContent = `# of Trajectories for stage ${i + 1}:`;

            const input = document.createElement('input');
            input.type = 'text';
            input.name = `trajectory_${i + 1}`;
            input.required = true;

            container.appendChild(label);
            container.appendChild(input);
            container.appendChild(document.createElement('br'));
        }
    }
});

document.getElementById('simulation-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    // Show the result area and start the car animation
    const carTrack = document.querySelector('.slider .track');
    carTrack.style.animationPlayState = 'running';

    // Handle form data and submit the request to the server
    const formData = new FormData(event.target);
    const jsonData = {
        sampling_depth: formData.get('sampling_depth'),
        num_trajectories: formData.get('num_trajectories')
    };

    fetch('/update_config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jsonData)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').textContent = data.output || 'Simulation completed.';

        // Pause the animation after any output is received
        carTrack.style.animationPlayState = 'paused';
    })
    .catch(error => {
        document.getElementById('result').textContent = 'Error: ' + error;

        carTrack.style.animationPlayState = 'paused';
    });
});