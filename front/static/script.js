document.getElementById('simulation-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const parameter = document.getElementById('parameter').value;
    const value = document.getElementById('value').value;

    fetch('/run_simulation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ parameter: parameter, value: value })
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.textContent = `Output: ${data.output}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
