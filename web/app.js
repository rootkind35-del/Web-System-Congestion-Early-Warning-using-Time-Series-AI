// Initialize Chart.js
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = 'Inter';

const ctx = document.getElementById('liveChart').getContext('2d');
const liveChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Actual CPU Load',
                borderColor: '#e2e8f0',
                backgroundColor: 'rgba(226, 232, 240, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                data: [],
                tension: 0.4
            },
            {
                label: 'Predicted T+5m',
                borderColor: '#f97316',
                borderDash: [5, 5],
                borderWidth: 2,
                pointRadius: 0,
                data: [],
                tension: 0.4
            },
            {
                label: 'EMA Dynamic Threshold',
                borderColor: '#3b82f6',
                borderWidth: 1.5,
                pointRadius: 0,
                data: [],
                tension: 0.4
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0 // Turn off animation for smooth streaming
        },
        scales: {
            x: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' }
            },
            y: {
                min: 0,
                max: 120,
                grid: { color: 'rgba(255, 255, 255, 0.05)' }
            }
        },
        plugins: {
            legend: {
                position: 'top',
            }
        }
    }
});

// DOM Elements
const elMetricCpu = document.getElementById('metric-cpu');
const elMetricThresh = document.getElementById('metric-thresh');
const elMetricInfer = document.getElementById('metric-infer');
const elStatusT5 = document.getElementById('status-t5');
const elStatusDrift = document.getElementById('status-drift');
const elLogConsole = document.getElementById('log-console');

const MAX_POINTS = 60; // Keep last 60 minutes on chart

function addLog(msg, type = 'normal') {
    const p = document.createElement('p');
    const time = new Date().toLocaleTimeString();
    p.innerHTML = `[${time}] ${msg}`;
    
    if (type === 'alert') p.className = 'log-alert';
    else if (type === 'warning') p.className = 'log-warning';
    else if (type === 'sys') p.className = 'sys-msg';
    else p.className = 'log-normal';
    
    elLogConsole.appendChild(p);
    elLogConsole.scrollTop = elLogConsole.scrollHeight;
}

// Connect to SSE Stream
const evtSource = new EventSource("/stream");

evtSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    // Update Metrics
    elMetricCpu.innerText = `${data.actual_cpu.toFixed(1)} %`;
    elMetricThresh.innerText = `${data.ema_threshold.toFixed(1)} %`;
    elMetricInfer.innerText = `${data.infer_time.toFixed(1)} ms`;
    
    // Update Chart
    const timeLabel = data.time;
    if (liveChart.data.labels.length >= MAX_POINTS) {
        liveChart.data.labels.shift();
        liveChart.data.datasets.forEach(ds => ds.data.shift());
    }
    
    liveChart.data.labels.push(timeLabel);
    liveChart.data.datasets[0].data.push(data.actual_cpu);
    liveChart.data.datasets[1].data.push(data.pred_t5);
    liveChart.data.datasets[2].data.push(data.ema_threshold);
    liveChart.update();
    
    // Congestion Logic (T+5 Alert)
    if (data.alert_t5) {
        elStatusT5.innerHTML = `<div class="dot red"></div> T+5 Status: DANGER`;
        elStatusT5.style.borderColor = "rgba(239, 68, 68, 0.5)";
        if (Math.random() > 0.8) {
            addLog(`WARNING: AI predicted CPU spike at T+5 (${data.pred_t5.toFixed(1)}% > ${data.ema_threshold.toFixed(1)}%). Auto-scaling Triggered!`, 'alert');
        }
    } else {
        elStatusT5.innerHTML = `<div class="dot green"></div> T+5 Status: Safe`;
        elStatusT5.style.borderColor = "rgba(255, 255, 255, 0.08)";
    }
    
    // Drift Logic
    if (data.drift) {
        elStatusDrift.innerHTML = `<div class="dot red"></div> Model Drift: DETECTED`;
        addLog(`CONCEPT DRIFT DETECTED. The model's error is unusually high. Retraining pipeline recommended.`, 'warning');
    }
};

evtSource.onerror = function(err) {
    console.error("SSE Error:", err);
    addLog("Connection lost. Reconnecting...", 'sys');
};
