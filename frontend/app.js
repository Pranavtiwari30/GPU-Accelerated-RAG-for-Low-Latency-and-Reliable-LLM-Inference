const API_BASE = "http://127.0.0.1:8000/api";

// Elements
const form = document.getElementById("query-form");
const submitBtn = document.getElementById("submit-btn");
const submitLoader = document.getElementById("submit-loader");
const btnText = document.querySelector(".btn-text");

const emptyState = document.getElementById("empty-state");
const resultContent = document.getElementById("result-content");

const answerText = document.getElementById("answer-text");
const resLatency = document.getElementById("res-latency");
const resTokens = document.getElementById("res-tokens");
const pipelineList = document.getElementById("pipeline-list");
const docsList = document.getElementById("docs-list");

let benchmarkChartInstance = null;
let hallucinationChartInstance = null;
let modelLatencies = { vanilla: 0, cpu_rag: 0, gpu_rag: 0 };

// Initialization
document.addEventListener("DOMContentLoaded", () => {
    fetchSystemStats();
    fetchBenchmarks();
    renderLatencyChart();
    
    // Poll stats every 5 seconds
    setInterval(fetchSystemStats, 5000);

    // Sidebar navigation click handler
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active from all tabs
            document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // Hide all views
            document.querySelectorAll('.view-section').forEach(view => view.classList.remove('active'));
            
            // Show target view
            const targetId = item.getAttribute('data-target');
            if (targetId) {
                document.getElementById(targetId).classList.add('active');
            }
            
            // Update Topbar Title
            document.getElementById('page-title').textContent = item.textContent.trim();
        });
    });
});

// Handle Form Submission
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const query = document.getElementById("query-input").value.trim();
    const model = document.getElementById("model-select").value;
    const dataset = document.getElementById("dataset-select").value;
    
    if (!query) return;

    // Loading State
    btnText.classList.add("hidden");
    submitLoader.classList.remove("hidden");
    submitBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/query?q=${encodeURIComponent(query)}&model=${model}&dataset=${dataset}`);
        const data = await response.json();
        
        if (data.error) {
            alert("API Error: " + data.error);
        } else {
            renderResults(data);
            modelLatencies[model] = data.metrics.total_ms;
            renderLatencyChart();
        }
    } catch (error) {
        console.error("Error fetching query:", error);
        alert("Failed to connect to the backend. Is it running?");
    } finally {
        // Reset Loading State
        btnText.classList.remove("hidden");
        submitLoader.classList.add("hidden");
        submitBtn.disabled = false;
        
        // Refresh stats after query
        fetchSystemStats();
    }
});

function renderResults(data) {
    emptyState.classList.add("hidden");
    resultContent.classList.remove("hidden");
    
    // Animate answer
    answerText.textContent = data.response || "No response generated.";
    
    // Metrics
    resLatency.textContent = `${data.metrics.total_ms.toFixed(1)} ms`;
    resTokens.textContent = data.metrics.tokens || "N/A";
    
    // Pipeline
    pipelineList.innerHTML = "";
    if (data.pipeline && data.pipeline.length > 0) {
        data.pipeline.forEach(step => {
            const li = document.createElement("li");
            
            const successIcon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
            const pendingIcon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>`;
            const icon = step.status === "success" ? successIcon : pendingIcon;
            
            li.innerHTML = `
                <div class="step-name">
                    <span>${icon}</span>
                    <span>${step.step}</span>
                </div>
                <div class="step-status">${step.detail}</div>
            `;
            pipelineList.appendChild(li);
        });
    }

    // Docs
    docsList.innerHTML = "";
    if (data.retrieved_docs && data.retrieved_docs.length > 0) {
        data.retrieved_docs.forEach((doc, idx) => {
            const div = document.createElement("div");
            div.className = "doc-item";
            div.textContent = `[Doc ${idx + 1}] ${doc.text.substring(0, 150)}...`;
            docsList.appendChild(div);
        });
    } else {
        docsList.innerHTML = "<div class='doc-item'>No documents retrieved.</div>";
    }
}

async function fetchSystemStats() {
    try {
        // Fetch GPU
        const gpuRes = await fetch(`${API_BASE}/gpu`);
        if (gpuRes.ok) {
            const gpuData = await gpuRes.json();
            
            const vramPct = (gpuData.memory_used / gpuData.memory_total) * 100;
            document.getElementById("vram-fill").style.width = `${vramPct}%`;
            document.getElementById("vram-used").textContent = gpuData.memory_used;
            document.getElementById("vram-total").textContent = gpuData.memory_total;
            
            const tempEl = document.getElementById("gpu-temp");
            tempEl.textContent = `${gpuData.temperature}°C`;
            if (gpuData.temperature > 80) tempEl.style.color = "var(--danger)";
            else if (gpuData.temperature > 65) tempEl.style.color = "orange";
            else tempEl.style.color = "var(--success)";
            
            updateSystemStatus("online");
        }
        
        // Fetch Metrics
        const metricsRes = await fetch(`${API_BASE}/metrics`);
        if (metricsRes.ok) {
            const metricsData = await metricsRes.json();
            document.getElementById("total-queries").textContent = metricsData.total_queries;
            document.getElementById("avg-latency").textContent = `${metricsData.avg_latency}ms`;
            document.getElementById("throughput").textContent = `${metricsData.tokens_per_sec} t/s`;
        }
        
    } catch (e) {
        console.log("Backend might be offline.");
        updateSystemStatus("offline");
    }
}

function updateSystemStatus(status) {
    const dot = document.querySelector(".status-dot");
    const text = document.getElementById("system-status-text");
    
    if (status === "online") {
        dot.className = "status-dot online";
        text.textContent = "System Online";
    } else {
        dot.className = "status-dot offline";
        text.textContent = "System Offline";
    }
}

function renderLatencyChart() {
    const ctx = document.getElementById('benchmarkChart').getContext('2d');
    
    const labels = ['1 Vanilla LLM', '2 CPU RAG', '3 GPU RAG'];
    const latencyData = [modelLatencies.vanilla, modelLatencies.cpu_rag, modelLatencies.gpu_rag];
    
    if (benchmarkChartInstance) {
        benchmarkChartInstance.destroy();
    }
    
    Chart.defaults.color = '#8b8b8b';
    Chart.defaults.font.family = "'Inter', sans-serif";

    benchmarkChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Current Run Latency (ms)',
                data: latencyData,
                backgroundColor: [
                    'rgba(239, 68, 68, 0.7)',
                    'rgba(245, 158, 11, 0.7)',
                    'rgba(16, 185, 129, 0.7)'
                ],
                borderColor: [
                    'rgb(239, 68, 68)',
                    'rgb(245, 158, 11)',
                    'rgb(16, 185, 129)'
                ],
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

async function fetchBenchmarks() {
    try {
        const res = await fetch(`${API_BASE}/benchmarks`);
        if (res.ok) {
            const data = await res.json();
            renderHallucinationChart(data);
        }
    } catch (e) {
        console.error("Could not fetch benchmarks", e);
    }
}

function renderHallucinationChart(data) {
    const ctx = document.getElementById('hallucinationChart').getContext('2d');
    
    const labels = ['1 Vanilla LLM', '2 CPU RAG', '3 GPU RAG'];
    
    // Sort the data or match it up manually to ensure it's in the same order
    // 'data' might be dynamic based on the CSV, but we expect it to have hallucination_rate
    const halluMap = {};
    data.forEach(d => {
        if(d.system.toLowerCase().includes('vanilla')) halluMap.vanilla = d.hallucination_rate * 100;
        if(d.system.toLowerCase().includes('cpu')) halluMap.cpu_rag = d.hallucination_rate * 100;
        if(d.system.toLowerCase().includes('gpu')) halluMap.gpu_rag = d.hallucination_rate * 100;
    });

    const hallucinationData = [
        halluMap.vanilla || 0, 
        halluMap.cpu_rag || 0, 
        halluMap.gpu_rag || 0
    ];
    
    if (hallucinationChartInstance) {
        hallucinationChartInstance.destroy();
    }

    hallucinationChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Hallucination Rate (%)',
                data: hallucinationData,
                backgroundColor: [
                    'rgba(239, 68, 68, 0.7)',
                    'rgba(245, 158, 11, 0.7)',
                    'rgba(16, 185, 129, 0.7)'
                ],
                borderColor: [
                    'rgb(239, 68, 68)',
                    'rgb(245, 158, 11)',
                    'rgb(16, 185, 129)'
                ],
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}
