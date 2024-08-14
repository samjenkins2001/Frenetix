document.addEventListener('DOMContentLoaded', () => {
    const samplingDepthInput = document.getElementById('sampling_depth');
    const enableNumTrajectoriesCheckbox = document.getElementById('enable_num_trajectories');
    const enableTrajectorySpacingCheckbox = document.getElementById('enable_trajectory_spacing');
    const numTrajectoriesContainer = document.getElementById('num-trajectories-container');
    const trajectorySpacingContainer = document.getElementById('trajectory-spacing-container');
    const simulationForm = document.getElementById('simulation-form');
  
    // Function to generate input boxes based on sampling depth
    function generateInputs() {
      const depth = parseInt(samplingDepthInput.value, 10);
  
      // Clear existing inputs
      numTrajectoriesContainer.innerHTML = '';
      trajectorySpacingContainer.innerHTML = '';

      const existingHr = document.getElementById('separator-line');
        if (existingHr) {
            existingHr.remove();
        }
  
      // Generate Number of Trajectories inputs if enabled
      if (enableNumTrajectoriesCheckbox.checked) {
        for (let i = 0; i < depth; i++) {
          const label = document.createElement('label');
          label.textContent = `Number of Trajectories for Stage ${i + 1}:`;
          label.setAttribute('for', `num_trajectories_${i + 1}`);
  
          const input = document.createElement('input');
          input.type = 'number';
          input.id = `num_trajectories_${i + 1}`;
          input.name = `num_trajectories_${i + 1}`;
          input.min = '1';
          input.required = true;
  
          numTrajectoriesContainer.appendChild(label);
          numTrajectoriesContainer.appendChild(input);
          numTrajectoriesContainer.appendChild(document.createElement('br'));
        }
        numTrajectoriesContainer.style.display = 'block';
      } else {
        numTrajectoriesContainer.style.display = 'none';
      }
  
      // Generate Trajectory Spacing inputs if enabled
      if (enableTrajectorySpacingCheckbox.checked) {
        if (enableNumTrajectoriesCheckbox.checked) {
          // Create and insert the horizontal line if both checkboxes are checked
          const hr = document.createElement('hr');
          hr.id = 'separator-line';
          hr.style.margin = '20px 0'; // Optional: Add margin for spacing
          hr.style.color = '#007bff'
          numTrajectoriesContainer.parentNode.insertBefore(hr, trajectorySpacingContainer);
        }
        for (let i = 0; i < depth; i++) {
          const label = document.createElement('label');
          label.textContent = `Trajectory Spacing for Stage ${i + 1}:`;
          label.setAttribute('for', `trajectory_spacing_${i + 1}`);
  
          const input = document.createElement('input');
          input.type = 'number';
          input.id = `trajectory_spacing_${i + 1}`;
          input.name = `trajectory_spacing_${i + 1}`;
          input.min = '0';
          input.step = '0.01';
          input.required = true;
  
          trajectorySpacingContainer.appendChild(label);
          trajectorySpacingContainer.appendChild(input);
          trajectorySpacingContainer.appendChild(document.createElement('br'));
        }
        trajectorySpacingContainer.style.display = 'block';
      } else {
        trajectorySpacingContainer.style.display = 'none';
      }
    }
  
    // Event Listeners
    samplingDepthInput.addEventListener('input', () => {
      if (samplingDepthInput.value && parseInt(samplingDepthInput.value, 10) > 0) {
        generateInputs();
      } else {
        // Clear containers if invalid depth
        numTrajectoriesContainer.innerHTML = '';
        trajectorySpacingContainer.innerHTML = '';
        numTrajectoriesContainer.style.display = 'none';
        trajectorySpacingContainer.style.display = 'none';
      }
    });
  
    enableNumTrajectoriesCheckbox.addEventListener('change', generateInputs);
    enableTrajectorySpacingCheckbox.addEventListener('change', generateInputs);
  
    simulationForm.addEventListener('submit', (event) => {
      event.preventDefault(); // Prevent default form submission
  
      // Start the car animation
      const carTrack = document.querySelector('.slider .track');
      carTrack.style.animationPlayState = 'running';
  
      // Gather form data
      const formData = new FormData(simulationForm);
      const samplingDepth = parseInt(formData.get('sampling_depth'), 10);
  
      const spacingTraj = [];
      const numTrajectories = [];
      const trajectorySpacing = [];
  
      if (enableNumTrajectoriesCheckbox.checked) {
        spacingTraj.push(true);
        for (let i = 0; i < samplingDepth; i++) {
          const value = formData.get(`num_trajectories_${i + 1}`);
          if (!value) {
            alert(`Please enter Number of Trajectories for Stage ${i + 1}`);
            return;
          }
          numTrajectories.push(parseInt(value, 10));
        }
      } else {
        spacingTraj.push(false);
      }
  
      if (enableTrajectorySpacingCheckbox.checked) {
        spacingTraj.push(true);
        for (let i = 0; i < samplingDepth; i++) {
          const value = formData.get(`trajectory_spacing_${i + 1}`);
          if (!value) {
            alert(`Please enter Trajectory Spacing for Stage ${i + 1}`);
            return;
          }
          trajectorySpacing.push(parseFloat(value));
        }
      } else {
        spacingTraj.push(false);
      }
  
      const jsonData = {
        sampling_depth: samplingDepth,
        spacing_traj: spacingTraj,
        num_trajectories: numTrajectories,
        trajectory_spacing: trajectorySpacing
      };
  
      // Send data to the server
      fetch('/update_config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jsonData)
      })
        .then(response => response.json())
        .then(data => {
          document.getElementById('result').textContent = data.output || data.message || 'Simulation completed.';
  
          // Pause the animation after receiving a response
          carTrack.style.animationPlayState = 'paused';
        })
        .catch(error => {
          document.getElementById('result').textContent = 'Error: ' + error;
          carTrack.style.animationPlayState = 'paused';
        });
    });
  });
  