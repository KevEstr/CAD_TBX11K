// --- State ---
let state = 'IDLE';
let sessionId = null;
let currentOverlay = 'regions';
let baseImageB64 = null;
let regionImageB64 = null;
let occlusionB64 = null;
let currentFile = null;

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

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', (e) => { e.preventDefault(); uploadZone.classList.remove('dragover'); handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });
btnReset.addEventListener('click', resetUI);

function handleFile(file) {
  if (!file) return;
  if (file.size > 20 * 1024 * 1024) { showError('Archivo demasiado grande. Máximo 20MB.'); return; }
  const parts = file.name.split('.');
  const ext = '.' + parts[parts.length - 1].toLowerCase();
  if (['.png', '.jpg', '.jpeg'].indexOf(ext) === -1) { showError('Formato no soportado. Usa PNG o JPG.'); return; }
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
  baseImageB64 = null;
  regionImageB64 = null;
  uploadZone.hidden = false;
  previewSection.hidden = true;
  loadingSection.hidden = true;
  resultsSection.hidden = true;
  errorSection.hidden = true;
  progressCard.hidden = true;
  progressFill.style.width = '0%';
  overlaySelect.value = 'regions';
  overlaySelect.options[2].disabled = true;
  fileInput.value = '';
}

btnAnalyze.addEventListener('click', async () => {
  if (!currentFile) return;
  setState('WAITING');
  loadingSection.hidden = false;
  resultsSection.hidden = true;
  errorSection.hidden = true;

  const form = new FormData();
  form.append('file', currentFile);

  try {
    const resp = await fetch('/api/v1/predict', { method: 'POST', body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    renderResults(data);
    sessionId = data.session_id;
    loadingSection.hidden = true;
    resultsSection.hidden = false;
    startOcclusionPolling(data.session_id);
  } catch (e) {
    showError('Error al analizar: ' + e.message);
  }
});

function renderResults(data) {
  const cls = data.prediction.class;
  const diagEl = document.getElementById('diagnosis-class');
  const label = cls === 'tb' ? 'TB PROBABLE' : 'NO TB';
  diagEl.textContent = label + ' (' + (data.prediction.confidence * 100).toFixed(1) + '%)';
  diagEl.className = 'diagnosis-class ' + cls;

  const notb = data.probabilities.no_tb;
  const tb = data.probabilities.tb;
  document.getElementById('bar-notb').style.width = (notb * 100) + '%';
  document.getElementById('bar-tb').style.width = (tb * 100) + '%';
  document.getElementById('val-notb').textContent = (notb * 100).toFixed(1) + '%';
  document.getElementById('val-tb').textContent = (tb * 100).toFixed(1) + '%';

  document.getElementById('model-meta').textContent =
    'Umbral ' + data.prediction.threshold.toFixed(3) + ' | ' + data.metadata.inference_time_ms + 'ms | ' + data.metadata.model;

  baseImageB64 = data.image_base64;

  if (data.level1) {
    const rc = document.getElementById('regions-list');
    rc.innerHTML = '';
    const vals = Object.values(data.level1.regions);
    const maxVal = Math.max.apply(null, vals);
    for (const name of Object.keys(data.level1.regions)) {
      const val = data.level1.regions[name];
      const widthPct = maxVal > 0 ? (val / maxVal * 100) : 0;
      rc.innerHTML += '<div class="region-item"><span class="region-name">' + name +
        '</span><div class="region-bar"><div class="region-fill" style="width:' + widthPct +
        '%"></div></div><span class="region-val">' + (val * 100).toFixed(0) + '%</span></div>';
    }

    const fc = document.getElementById('features-list');
    fc.innerHTML = '';
    for (const ft of data.level1.top_features) {
      const dir = ft.direction === 'aumenta TB' ? '+' : '−';
      fc.innerHTML += '<div class="feature-item"><span class="fi-name">' + escapeHtml(ft.name) +
        '</span><span class="fi-group">' + ft.group + '</span><span class="fi-val">' + dir +
        ft.importance.toFixed(3) + '</span></div>';
    }

    const fam = document.getElementById('families-list');
    fam.innerHTML = '';
    for (const name of Object.keys(data.level1.families)) {
      const val = data.level1.families[name];
      fam.innerHTML += '<div class="family-item"><span class="fm-name">' + name +
        '</span><div class="fm-bar"><div class="fm-fill" style="width:' + (val * 100) +
        '%"></div></div><span class="fm-val">' + (val * 100).toFixed(1) + '%</span></div>';
    }

    regionImageB64 = data.level1.region_heatmap_base64;
  }

  currentOverlay = 'regions';
  overlaySelect.value = 'regions';
  drawOverlay('regions');
}

function drawOverlay(mode) {
  if (!baseImageB64) return;
  let src = baseImageB64;
  if (mode === 'regions' && regionImageB64) src = regionImageB64;
  else if (mode === 'occlusion' && occlusionB64) src = occlusionB64;

  const img = new Image();
  img.onload = () => {
    canvasCtx.clearRect(0, 0, 512, 512);
    canvasCtx.drawImage(img, 0, 0, 512, 512);
  };
  img.src = src;
}

overlaySelect.addEventListener('change', () => {
  currentOverlay = overlaySelect.value;
  drawOverlay(currentOverlay);
});

// --- Occlusion polling ---
function startOcclusionPolling(sid) {
  progressCard.hidden = false;
  progressFill.style.width = '0%';
  progressText.textContent = 'Generando mapa de oclusión...';

  fetch('/api/v1/predict/' + sid + '/occlude', { method: 'POST' }).catch(() => {});

  const interval = setInterval(async () => {
    try {
      const resp = await fetch('/api/v1/predict/' + sid + '/progress');
      const data = await resp.json();
      if (data.status === 'complete') {
        clearInterval(interval);
        onOcclusionComplete(data.result);
        cleanupSession();
      } else if (data.status === 'processing' && data.progress) {
        const p = data.progress;
        progressFill.style.width = ((p.current / p.total) * 100) + '%';
        progressText.textContent = p.detail + ' (' + p.current + '/' + p.total + ')';
      }
    } catch (e) { /* retry next tick */ }
  }, 1500);
}

function onOcclusionComplete(res) {
  progressFill.style.width = '100%';
  progressText.textContent = 'Mapa de oclusión generado (' + (res.time_ms / 1000).toFixed(1) + 's)';
  if (res.overlay) {
    occlusionB64 = res.overlay;
    overlaySelect.options[2].disabled = false;
    overlaySelect.value = 'occlusion';
    currentOverlay = 'occlusion';
    drawOverlay('occlusion');
  }
  setState('COMPLETE');
  cleanupSession();
}

function cleanupSession() {
  if (sessionId) {
    fetch('/api/v1/session/' + sessionId, { method: 'DELETE' }).catch(() => {});
  }
}

function setState(s) { state = s; }
function escapeHtml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

function showError(msg) {
  errorMessage.textContent = msg;
  loadingSection.hidden = true;
  resultsSection.hidden = true;
  errorSection.hidden = false;
  setState('IDLE');
}