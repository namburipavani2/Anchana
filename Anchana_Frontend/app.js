// ⚡ POINT DIRECTLY TO YOUR SUCCESSFUL LIVE RENDER BACKEND
const BACKEND_URL = "https://anchana.onrender.com";

let uploadedFile = null;

// Event Listener for File Uploads
document.getElementById('fileInput').addEventListener('change', async (e) => {
    uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    document.getElementById('statusMessage').innerText = "Profiling incoming dataset metadata parameters...";
    
    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
        const response = await fetch(`${BACKEND_URL}/analyze-file`, { method: "POST", body: formData });
        if (!response.ok) throw new Error("Metadata calculation failure.");
        
        const data = await response.json();
        
        // Populate Target Dropdown Select Option Primitives
        const targetSelect = document.getElementById('targetSelect');
        targetSelect.innerHTML = "";
        data.columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.innerText = col;
            targetSelect.appendChild(opt);
        });

        document.getElementById('configSection').style.display = "block";
        document.getElementById('statusMessage').innerText = `Dataset Profile Loaded successfully: ${data.shape[0]} rows × ${data.shape[1]} columns.`;
    } catch (err) {
        document.getElementById('statusMessage').innerText = `Backend node processing error: ${err.message}`;
    }
});

// Event Listener for Compute Pipeline Execution
document.getElementById('executeBtn').addEventListener('click', async () => {
    if (!uploadedFile) return;

    document.getElementById('statusMessage').innerText = "Training processing workflows across continuous backend instances...";
    document.getElementById('resultsDashboard').style.display = "none";

    const formData = new FormData();
    formData.append("file", uploadedFile);
    formData.append("problem_type", document.getElementById('trackSelect').value);
    formData.append("target_column", document.getElementById('targetSelect').value);
    formData.append("drop_columns", JSON.stringify([]));

    try {
        const response = await fetch(`${BACKEND_URL}/evaluate`, { method: "POST", body: formData });
        if (!response.ok) throw new Error("Computational Evaluation Failure.");

        const data = await response.json();
        
        // Render KPI Block
        document.getElementById('runtimeKPI').innerText = `${data.total_pipeline_time_sec} Sec`;
        
        // Build Evaluation Score & Metric Data Table Arrays
        renderTable(data.results);

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
    models.forEach(model => {
        headerRow.innerHTML += `<th>${model}</th>`;
    });

    const metrics = Object.keys(results[models[0]]);
    metrics.forEach(metric => {
        let rowHtml = `<tr><td><strong>${metric}</strong></td>`;
        models.forEach(model => {
            rowHtml += `<td>${results[model][metric]}</td>`;
        });
        rowHtml += "</tr>";
        bodyRows.innerHTML += rowHtml;
    });
}
