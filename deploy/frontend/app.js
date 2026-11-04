// --- State ---
let state = 'IDLE';
let sessionId = null;
let ws = null;
let currentOverlay = 'hog';
let preprocessedB64 = null;
let occlusionB64 = null;
let hogB64 = null;
let currentFile = null;

// --- DOM refs ---
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImg = document.getElementById('preview-img');
const previewFilename = document.getElementById('preview-filename');
const previewSize = document.getElementById('preview-size');
const btnAnalyze = document.getElementById('btn-analyze');
const btnReset = document.getElementById('btn-reset');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');
const progressCard = document.getElementById('progress-card');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const overlaySelect = document.getElementById('overlay-select');
const imageCanvas = document.getElementById('image-canvas');
const canvasCtx = imageCanvas.getContext('2d');

// --- File upload ---
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', (e) => { e.preventDefault(); uploadZone.classList.remove('dragover'); handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });
btnReset.addEventListener('click', resetUI);

function handleFile(file) {
  if (!file) return;
  const maxSize = 20 * 1024 * 1024;
  if (file.size > maxSize) { showError('Archivo demasiado grande. Máximo 20MB.'); return; }
  const parts = file.name.split('.');
  const ext = '.' + parts[parts.length - 1].toLowerCase();
  if (!['.png', '.jpg', '.jpeg'].includes(ext)) { showError('Formato no soportado. Usa PNG o JPG.'); return; }

  previewImg.src = URL.createObjectURL(file);
  previewFilename.textContent = file.name;
  previewSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
  uploadZone.hidden = true;
  previewSection.hidden = false;
  currentFile = file;
  state = 'SENDING';
}

function resetUI() {
  state = 'IDLE';
  sessionId = null;
  currentFile = null;
  occlusionB64 = null;
  preprocessedB64 = null;
  if (ws) { try { ws.close(); } catch(e) {} ws = null; }

  uploadZone.hidden = false;
  previewSection.hidden = true;
  loadingSection.hidden = true;
  resultsSection.hidden = true;
  errorSection.hidden = true;
  progressCard.hidden = true;
  progressFill.style.width = '0%';
  overlaySelect.value = 'hog';
  overlaySelect.options[2].disabled = true;
  overlaySelect.options[2].text = 'Oclusión (cargando...)';
  fileInput.value = '';
}

// --- Analyze ---
btnAnalyze.addEventListener('click', async () => {
  if (!currentFile) return;
  setState('WAITING_N1N2');
  loadingSection.hidden = false;
  resultsSection.hidden = true;
  errorSection.hidden = true;

  const form = new FormData();
  form.append('file', currentFile);

  try {
    const resp = await fetch('/api/v1/predict', { method: 'POST', body: form });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    renderResults(data);
    sessionId = data.session_id;
    loadingSection.hidden = true;
    resultsSection.hidden = false;
    setState('RESULTS_PARTIAL');
    connectWebSocket();
  } catch (e) {
    showError('Error al analizar: ' + e.message);
  }
});

// --- Render N1 + N2 ---
function renderResults(data) {
  // Diagnosis
  const cls = data.prediction.class;
  const diagEl = document.getElementById('diagnosis-class');
  diagEl.textContent = cls.toUpperCase() + ' (' + (data.prediction.confidence * 100).toFixed(1) + '%)';
  diagEl.className = 'diagnosis-class ' + cls;

  // Probabilities
  document.getElementById('bar-health').style.width = (data.probabilities.health * 100) + '%';
  document.getElementById('bar-sick').style.width = (data.probabilities.sick * 100) + '%';
  document.getElementById('bar-tb').style.width = (data.probabilities.tb * 100) + '%';
  document.getElementById('val-health').textContent = (data.probabilities.health * 100).toFixed(1) + '%';
  document.getElementById('val-sick').textContent = (data.probabilities.sick * 100).toFixed(1) + '%';
  document.getElementById('val-tb').textContent = (data.probabilities.tb * 100).toFixed(1) + '%';

  // Metadata
  document.getElementById('model-meta').textContent = 'Modelo: ' + data.metadata.model + ' | ' + data.metadata.inference_time_ms + 'ms';

  // Level 1: Regions
  if (data.level1) {
    const regionsContainer = document.getElementById('regions-list');
    regionsContainer.innerHTML = '';
    const maxVal = Math.max(...Object.values(data.level1.regions));
    for (const [name, val] of Object.entries(data.level1.regions)) {
      const pct = val * 100;
      const widthPct = maxVal > 0 ? (val / maxVal * 100) : 0;
      regionsContainer.innerHTML +=
        `<div class="region-item"><span class="region-name">${name}</span><div class="region-bar"><div class="region-fill" style="width:${widthPct}%"></div></div><span class="region-val">${pct.toFixed(0)}%</span></div>`;
    }

    // Top features
    const featuresContainer = document.getElementById('features-list');
    featuresContainer.innerHTML = '';
    if (data.level1.top_features) {
      const top10 = data.level1.top_features.slice(0, 10);
      for (const ft of top10) {
        featuresContainer.innerHTML +=
          `<div class="feature-item"><span class="fi-name">${escapeHtml(ft.name)}</span><span class="fi-group">${ft.group}</span><span class="fi-val">${ft.importance.toFixed(5)}</span></div>`;
      }
    }

    // Families
    const familiesContainer = document.getElementById('families-list');
    familiesContainer.innerHTML = '';
    if (data.level1.families) {
      for (const [name, val] of Object.entries(data.level1.families)) {
        const pct = val * 100;
        familiesContainer.innerHTML +=
          `<div class="family-item"><span class="fm-name">${name}</span><div class="fm-bar"><div class="fm-fill" style="width:${pct}%"></div></div><span class="fm-val">${pct.toFixed(1)}%</span></div>`;
      }
    }
  }

  // Level 2: HOG overlay
  if (data.level2 && data.level2.hog_overlay_base64) {
    hogB64 = data.level2.hog_overlay_base64;
    overlaySelect.options[1].disabled = false;
    currentOverlay = 'hog';
    overlaySelect.value = 'hog';
  }

  // Draw image
  preprocessedB64 = data.level1?.region_heatmap_base64;
  drawOverlay('none');
}

// --- Canvas drawing ---
function drawOverlay(mode) {
  if (!preprocessedB64) return;
  const img = new Image();
  img.onload = () => {
    canvasCtx.clearRect(0, 0, 512, 512);
    canvasCtx.drawImage(img, 0, 0, 512, 512);

    if (mode === 'hog' && hogB64) {
      const overlay = new Image();
      overlay.onload = () => canvasCtx.drawImage(overlay, 0, 0, 512, 512);
      overlay.src = hogB64;
    } else if (mode === 'occlusion' && occlusionB64) {
      const overlay = new Image();
      overlay.onload = () => canvasCtx.drawImage(overlay, 0, 0, 512, 512);
      overlay.src = occlusionB64;
    }
  };
  img.src = preprocessedB64;
}

overlaySelect.addEventListener('change', () => {
  currentOverlay = overlaySelect.value;
  drawOverlay(currentOverlay);
});

// --- WebSocket for N3 ---
let wsRetries = 0;
let lastProgressCount = 0;
const TOTAL_OCCLUSION_CELLS = 36;

function connectWebSocket() {
  if (!sessionId) return;
  closeWebSocket();

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = proto + '//' + location.host + '/ws/' + sessionId;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    progressCard.hidden = false;
    setState('RESULTS_PARTIAL');
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'progress') {
      lastProgressCount = msg.current;
      const pct = msg.percent;
      progressFill.style.width = pct + '%';
      progressText.textContent = msg.detail + ' (' + msg.current + '/' + msg.total + ')';
    } else if (msg.type === 'complete') {
      onOcclusionComplete(msg);
    } else if (msg.type === 'error') {
      progressText.textContent = 'Error: ' + msg.detail;
      setState('RESULTS_PARTIAL');
    }
  };

  ws.onclose = () => {
    ws = null;
    // No mostrar como completo si no llego el mensaje con los datos
    handleWsFailure();
  };
}

function onOcclusionComplete(msg) {
  progressFill.style.width = '100%';
  const timeStr = msg.occlusion_time_ms ? ' (' + (msg.occlusion_time_ms / 1000).toFixed(1) + 's)' : '';
  progressText.textContent = 'Mapa de oclusión generado' + timeStr;
  if (msg.occlusion_overlay_base64) {
    occlusionB64 = msg.occlusion_overlay_base64;
    overlaySelect.options[2].disabled = false;
    overlaySelect.options[2].text = 'Oclusión';
    overlaySelect.value = 'occlusion';
    currentOverlay = 'occlusion';
    drawOverlay('occlusion');
  }
  setState('RESULTS_COMPLETE');
  cleanupSession();
}

function handleWsFailure() {
  if (lastProgressCount === 0 && wsRetries < 2) {
    wsRetries++;
    setTimeout(() => connectWebSocket(), 5000);
  } else {
    progressText.textContent = 'No se pudo generar el mapa detallado. Resultados parciales disponibles.';
  }
}

function closeWebSocket() {
  if (ws) {
    try { ws.close(); } catch(e) {}
    ws = null;
  }
}

function cleanupSession() {
  if (sessionId) {
    fetch('/api/v1/session/' + sessionId, { method: 'DELETE' }).catch(() => {});
  }
}

function setState(s) { state = s; }

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function showError(msg) {
  errorMessage.textContent = msg;
  loadingSection.hidden = true;
  resultsSection.hidden = true;
  errorSection.hidden = false;
  setState('IDLE');
}