// ⚡ POINT DIRECTLY TO YOUR SUCCESSFUL LIVE RENDER BACKEND
const BACKEND_URL = "https://onrender.com";

let uploadedFile = null;
let activePerformanceChart = null;
let activeSpeedChart = null;

// 🔐 Event Listener for Account Registration Operations
document.getElementById('registerBtn').addEventListener('click', async () => {
    const userInp = document.getElementById('newUsername').value;
    const passInp = document.getElementById('newPassword').value;

    if (!userInp || !passInp) {
        alert("Please fill out both the username and password fields.");
        return;
    }

    try {
        const response = await fetch(`${BACKEND_URL}/api/register`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username: userInp, password: passInp })
        });

        const data = await response.json();

        if (response.ok) {
            alert(`🎉 ${data.message}`);
            document.getElementById('authSection').style.display = "none";
            document.getElementById('workspacePanel').style.display = "block";
            document.getElementById('statusMessage').innerText = "Account verified. Please upload a file to initialize training parameters.";
        } else {
            alert(`⚠️ Error: ${data.detail || "Registration failed"}`);
        }
    } catch (err) {
        alert(`Could not connect to backend server: ${err.message}`);
    }
});

// 🔍 Event Listener for Dataset File Profiling Endpoint
document.getElementById('fileInput').addEventListener('change', async (e) => {
    uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    document.getElementById('statusMessage').innerText = "Profiling incoming dataset metadata parameters...";
    document.getElementById('datasetProfileSection').style.display = "none";
    
    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
        const response = await fetch(`${BACKEND_URL}/analyze-file`, { method: "POST", body: formData });
        if (!response.ok) throw new Error("Metadata profile extraction failed.");
        
        const data = await response.json();
        
        // ➕ RENDER METADATA PROFILE PARAMETERS ON THE SIDEBAR BLOCK
        const profileBox = document.getElementById('profileMetricsContent');
        profileBox.innerHTML = `
            <strong>Shape Matrix:</strong> ${data.shape[0]} rows × ${data.shape[1]} cols<br>
            <strong>Total Elements Size:</strong> ${data.size}<br>
            <strong>Numerical Parameters:</strong> ${data.numerical_features}<br>
            <strong>Categorical Parameters:</strong> ${data.categorical_features}<br>
            <strong>Missing Columns:</strong> ${data.null_features}<br>
            <strong>Duplicate Rows:</strong> ${data.duplicate_rows}
        `;
        document.getElementById('datasetProfileSection').style.display = "block";
        
        // Populate Target Dropdown Component Options
        const targetSelect = document.getElementById('targetSelect');
        targetSelect.innerHTML = "";
        data.columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.innerText = col;
            targetSelect.appendChild(opt);
        });

        document.getElementById('configSection').style.display = "block";
        document.getElementById('statusMessage').innerText = "Dataset profile metrics calculated successfully. Configure track variables and execute modeling.";
    } catch (err) {
        document.getElementById('statusMessage').innerText = `Backend node processing error: ${err.message}`;
    }
});

// ⚙️ Event Listener for Compute Pipeline Execution Route
document.getElementById('executeBtn').addEventListener('click', async () => {
    if (!uploadedFile) return;

    document.getElementById('statusMessage').innerText = "Training processing workflows across continuous backend instances...";
    document.getElementById('resultsDashboard').style.display = "none";

    const problemType = document.getElementById('trackSelect').value;
    const formData = new FormData();
    formData.append("file", uploadedFile);
    formData.append("problem_type", problemType);
    formData.append("target_column", document.getElementById('targetSelect').value);
    formData.append("drop_columns", JSON.stringify([]));

    try {
        const response = await fetch(`${BACKEND_URL}/evaluate`, { method: "POST", body: formData });
        if (!response.ok) throw new Error("Computational Evaluation Failure.");

        const data = await response.json();
        
        // Render Execution Time KPI Card Block
        document.getElementById('runtimeKPI').innerText = `${data.total_pipeline_time_sec} Sec`;
        
        // Build Evaluation Score & Metric Data Table Grid Arrays
        renderTable(data.results);
        
        // ➕ RENDER PERFORMANCE GRAPH VISUALIZATION ARRAYS VIA CHART.JS
        renderCharts(data.results, problemType);

        document.getElementById('statusMessage').innerText = "Execution completed successfully!";
        document.getElementById('resultsDashboard').style.display = "block";
    } catch (err) {
        document.getElementById('statusMessage').innerText = `Processing Error: ${err.message}`;
    }
});

function renderTable(results) {
    const headerRow = document.getElementById('tableHeader');
    const bodyRows = document.getElementById('tableBody');
    headerRow.innerHTML = "<th>Metrics</th>";
    bodyRows.innerHTML = "";

    const models = Object.keys(results);
    models.forEach(model => { headerRow.innerHTML += `<th>${model}</th>`; });

    const metrics = Object.keys(results[models[0]]);
    metrics.forEach(metric => {
        let rowHtml = `<tr><td><strong>${metric}</strong></td>`;
        models.forEach(model => { rowHtml += `<td>${results[model][metric]}</td>`; });
        rowHtml += "</tr>";
        bodyRows.innerHTML += rowHtml;
    });
}

// ➕ NEW DYNAMIC BAR GRAPH RENDERING INTERACTION CONTROLLER
function renderCharts(results, trackCode) {
    const models = Object.keys(results);
    const runtimes = models.map(m => parseFloat(results[m]['Execution_Time_Sec']));
    
    // Determine target accuracy splits depending on Supervised tracks
    const accuracyMetricName = (trackCode === 'R') ? 'Acc_Test_R2' : 'Acc_Test';
    const trainAccuracyScores = models.map(m => parseFloat(results[m]['Acc_Train'] || 0));
    const testAccuracyScores = models.map(m => parseFloat(results[m][accuracyMetricName] || 0));

    // Reset previous chart instances to clear old cache rendering overlaps
    if (activePerformanceChart) activePerformanceChart.destroy();
    if (activeSpeedChart) activeSpeedChart.destroy();

    // 1. Draw Performance Distributions Bar Graphs
    const ctxPerf = document.getElementById('performanceChart').getContext('2d');
    activePerformanceChart = new Chart(ctxPerf, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                { label: 'Train Score Accuracy', data: trainAccuracyScores, backgroundColor: '#388bfd' },
                { label: 'Test Split Evaluation Metric', data: testAccuracyScores, backgroundColor: '#56a2f3' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { min: 0, max: 1.05, grid: { color: '#30363d' } }, x: { grid: { display: false } } },
            plugins: { legend: { labels: { color: '#c9d1d9' } } }
        }
    });

    // 2. Draw Speed Latency Bar Graphs
    const ctxSpeed = document.getElementById('speedChart').getContext('2d');
    activeSpeedChart = new Chart(ctxSpeed, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [{ label: 'Execution Latency (Seconds)', data: runtimes, backgroundColor: '#ff4b4b' }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { grid: { color: '#30363d' } }, x: { grid: { display: false } } },
            plugins: { legend: { labels: { color: '#c9d1d9' } } }
        }
    });
}
