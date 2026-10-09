// ⚡ POINT DIRECTLY TO YOUR LIVE RENDER BACKEND NODE
const BACKEND_URL = "https://onrender.com";


let uploadedFile = null;
let isSignUpMode = true; // State tracker for form toggles
let activePerformanceChart = null;
let activeSpeedChart = null;

// Auth View Mode Interaction Toggle Switcher
document.getElementById('toggleAuthLink').addEventListener('click', () => {
    isSignUpMode = !isSignUpMode;
    const title = document.getElementById('authTitle');
    const submitBtn = document.getElementById('authSubmitBtn');
    const link = document.getElementById('toggleAuthLink');
    
    if (isSignUpMode) {
        title.innerText = "🔐 Create Account";
        submitBtn.innerText = "Register Profile Account";
        submitBtn.style.backgroundColor = "#58a6ff";
        link.innerText = "Already a member? Sign In instead";
    } else {
        title.innerText = "🔑 Member Sign In";
        submitBtn.innerText = "Sign In securely";
        submitBtn.style.backgroundColor = "#238636";
        link.innerText = "New member? Create an account instead";
    }
});

// Authentication Form Action API Pipeline Router
document.getElementById('authSubmitBtn').addEventListener('click', async () => {
    const userInp = document.getElementById('authUsername').value;
    const passInp = document.getElementById('authPassword').value;

    if (!userInp || !passInp) {
        alert("Please completely fill out both user credential fields.");
        return;
    }

    // Direct routing targeting endpoints dynamically depending on active state
    const targetEndpoint = isSignUpMode ? "/api/register" : "/api/login";

    try {
        const response = await fetch(`${BACKEND_URL}${targetEndpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username: userInp, password: passInp })
        });

        const data = await response.json();

        if (response.ok) {
            alert(`🎉 Access Granted: ${data.message || "Authorized successfully."}`);
            document.getElementById('authSection').style.display = "none";
            document.getElementById('workspacePanel').style.display = "block";
            document.getElementById('statusMessage').innerText = "Security verified. Drop a clean file (.csv, .xlsx) inside the workspace folder selector.";
        } else {
            alert(`⚠️ Access Denied: ${data.detail || "Authentication sequence mismatch."}`);
        }
    } catch (err) {
        alert(`Could not connect to backend authorization instances: ${err.message}`);
    }
});

// 🔍 DATASET STRUCTURAL METADATA METRICS PROFILER LISTENER
document.getElementById('fileInput').addEventListener('change', async (e) => {
    uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    document.getElementById('statusMessage').innerText = "Profiling incoming dataset metadata parameters via remote servers...";
    document.getElementById('datasetProfileSection').style.display = "none";
    
    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
        const response = await fetch(`${BACKEND_URL}/analyze-file`, { method: "POST", body: formData });
        if (!response.ok) throw new Error("Metadata profiling failed on computational node clusters.");
        
        const data = await response.json();
        
        // Render Dataset Metadata parameters onto the UI sidebar summary view container block
        const profileBox = document.getElementById('profileMetricsContent');
        profileBox.innerHTML = `
            <strong>Shape Matrix:</strong> ${data.shape[0]} rows × ${data.shape[1]} columns<br>
            <strong>Total Elements Size:</strong> ${data.size}<br>
            <strong>Numerical Parameters:</strong> ${data.numerical_features}<br>
            <strong>Categorical Parameters:</strong> ${data.categorical_features}<br>
            <strong>Missing Value Columns:</strong> ${data.null_features}<br>
            <strong>Identified Duplicate Rows:</strong> ${data.duplicate_rows}
        `;
        document.getElementById('datasetProfileSection').style.display = "block";
        
        // Populate operational target drop-down choices list arrays
        const targetSelect = document.getElementById('targetSelect');
        targetSelect.innerHTML = "";
        data.columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.innerText = col;
            targetSelect.appendChild(opt);
        });

        document.getElementById('configSection').style.display = "block";
        document.getElementById('statusMessage').innerText = "Data architecture matrices structured completely. Configure analytical track configurations.";
    } catch (err) {
        document.getElementById('statusMessage').innerText = `Data Profiling failure protocol triggered: ${err.message}`;
    }
});

// Compute Engine modeling pipeline execution handler
document.getElementById('executeBtn').addEventListener('click', async () => {
    if (!uploadedFile) return;

    document.getElementById('statusMessage').innerText = "Training processing workflows across isolated backend instances...";
    document.getElementById('resultsDashboard').style.display = "none";

    const problemType = document.getElementById('trackSelect').value;
    const formData = new FormData();
    formData.append("file", uploadedFile);
    formData.append("problem_type", problemType);
    formData.append("target_column", document.getElementById('targetSelect').value);
    formData.append("drop_columns", JSON.stringify([]));

    try {
        const response = await fetch(`${BACKEND_URL}/evaluate`, { method: "POST", body: formData });
        if (!response.ok) throw new Error("Modeling suite run crash encountered.");

        const data = await response.json();
        
        document.getElementById('runtimeKPI').innerText = `${data.total_pipeline_time_sec} Sec`;
        renderTable(data.results);
        
        // 📈 EXECUTE CANVAS RENDERING INTERACTION TASKS VIA CHART.JS
        renderCharts(data.results, problemType);

        document.getElementById('statusMessage').innerText = "Execution completed successfully!";
        document.getElementById('resultsDashboard').style.display = "block";
    } catch (err) {
        document.getElementById('statusMessage').innerText = `Computational Error: ${err.message}`;
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

// DYNAMIC BAR PLOTS SYSTEM INTERACTION LAYER
function renderCharts(results, trackCode) {
    const models = Object.keys(results);
    const runtimes = models.map(m => parseFloat(results[m]['Execution_Time_Sec']));

    const accuracyMetricName = (trackCode === 'R') ? 'Acc_Test_R2' : 'Acc_Test';
    const trainAccuracyScores = models.map(m => parseFloat(results[m]['Acc_Train'] || 0));
    const testAccuracyScores = models.map(m => parseFloat(results[m][accuracyMetricName] || 0));

    if (activePerformanceChart) activePerformanceChart.destroy();
    if (activeSpeedChart) activeSpeedChart.destroy();

    // 1. Plot Model Evaluation Split Scores
    const ctxPerf = document.getElementById('performanceChart').getContext('2d');
    activePerformanceChart = new Chart(ctxPerf, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                { label: 'Train Accuracy Score', data: trainAccuracyScores, backgroundColor: '#388bfd' },
                { label: 'Test Accuracy Score', data: testAccuracyScores, backgroundColor: '#56a2f3' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { min: 0, max: 1.05 } }
        }
    });

    // 2. Plot Model Latency Run Speed Metrics
    const ctxSpeed = document.getElementById('speedChart').getContext('2d');
    activeSpeedChart = new Chart(ctxSpeed, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [{ label: 'Execution Speed (Seconds)', data: runtimes, backgroundColor: '#ff4b4b' }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}
