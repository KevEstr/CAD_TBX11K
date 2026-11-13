let state = 'IDLE';
let sessionId = null;
let currentOverlay = 'regions';
let baseImageB64 = null;
let regionImageB64 = null;
let occlusionB64 = null;
let segmentationB64 = null;
let currentFile = null;
let occlusionInterval = null;
let occlusionStartedAt = null;
const OCCLUSION_TIMEOUT_MS = 120000;

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
const progressPercent = document.getElementById('progress-percent');
const progressText = document.getElementById('progress-text');
const overlaySelect = document.getElementById('overlay-select');
const overlayLegend = document.getElementById('overlay-legend');
const viewNote = document.getElementById('view-note');
const viewerTitle = document.getElementById('viewer-title');
const imageCanvas = document.getElementById('image-canvas');
const canvasCtx = imageCanvas.getContext('2d');

const classLabels = {
  health: 'Sin hallazgos',
  sick: 'Otra enfermedad',
  tb: 'Tuberculosis probable',
};

const classNotes = {
  health: 'La probabilidad de tuberculosis es baja: no supera el nivel que el modelo usa para alertar. El resultado corresponde a normalidad.',
  sick: 'La probabilidad de tuberculosis es baja: no supera el nivel que el modelo usa para alertar. El resultado corresponde a otra enfermedad.',
  tb: 'La probabilidad estimada de tuberculosis es alta: supera el nivel que el modelo usa para alertar.',
};

function friendlyFamily(name) {
  const key = String(name).toLowerCase();
  if (key.includes('hog')) return 'Formas y bordes';
  if (key.includes('textur')) return 'Textura';
  return 'Otras características';
}

function friendlyGroup(name) {
  const key = String(name).toLowerCase();
  if (key.includes('hog')) return 'Formas';
  if (key.includes('textur')) return 'Textura';
  return 'Otras';
}

const featureTerms = {
  contrast: 'contraste',
  entropy: 'entropía',
  homogeneity: 'homogeneidad',
  energy: 'energía',
  correlation: 'correlación',
  dissimilarity: 'disimilitud',
  mean: 'media',
  std: 'desviación',
  skew: 'asimetría',
  kurtosis: 'curtosis',
};

function friendlyFeature(name) {
  const raw = String(name);
  const lower = raw.toLowerCase();
  if (lower.startsWith('hog_')) {
    return 'Formas y bordes · ' + raw.slice(4).replace(/_/g, '-');
  }
  if (lower.startsWith('glcm_') || lower.startsWith('lbp_') || lower.startsWith('gabor_')) {
    const prefix = lower.startsWith('glcm_') ? 5 : (lower.startsWith('lbp_') ? 4 : 6);
    const suffix = raw.slice(prefix).split('_').map((part) => featureTerms[part] || part).join(' ');
    return 'Textura · ' + suffix;
  }
  if (lower.startsWith('intensity_')) {
    const suffix = raw.slice(10).split('_').map((part) => featureTerms[part] || part).join(' ');
    return 'Intensidad · ' + suffix;
  }
  return raw.replace(/_/g, ' ');
}

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});
uploadZone.addEventListener('dragover', (event) => {
  event.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', (event) => {
  event.preventDefault();
  uploadZone.classList.remove('dragover');
  handleFile(event.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});
btnReset.addEventListener('click', resetUI);

function handleFile(file) {
  if (!file) return;
  if (file.size > 20 * 1024 * 1024) {
    showError('Archivo demasiado grande. El máximo permitido es 20 MB.');
    return;
  }
  const parts = file.name.split('.');
  const ext = '.' + parts[parts.length - 1].toLowerCase();
  if (['.png', '.jpg', '.jpeg'].indexOf(ext) === -1) {
    showError('Formato no soportado. Utiliza una imagen PNG o JPG.');
    return;
  }
  if (previewImg.src.indexOf('blob:') === 0) {
    URL.revokeObjectURL(previewImg.src);
  }
  previewImg.src = URL.createObjectURL(file);
  previewImg.alt = 'Vista previa de ' + file.name;
  previewFilename.textContent = file.name;
  previewSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
  uploadZone.hidden = true;
  previewSection.hidden = false;
  currentFile = file;
  state = 'SENDING';
}

function setViewOption(value, disabled) {
  const option = overlaySelect.querySelector('option[value="' + value + '"]');
  if (option) option.disabled = disabled;
}

function resetUI() {
  stopOcclusionPolling();
  cleanupSession();
  state = 'IDLE';
  sessionId = null;
  currentFile = null;
  occlusionB64 = null;
  segmentationB64 = null;
  baseImageB64 = null;
  regionImageB64 = null;
  currentOverlay = 'regions';
  uploadZone.hidden = false;
  previewSection.hidden = true;
  loadingSection.hidden = true;
  resultsSection.hidden = true;
  errorSection.hidden = true;
  progressCard.hidden = true;
  progressFill.style.width = '0%';
  progressPercent.textContent = '0%';
  overlayLegend.hidden = true;
  overlaySelect.value = 'regions';
  setViewOption('segmentation', true);
  setViewOption('occlusion', true);
  fileInput.value = '';
  if (previewImg.src.indexOf('blob:') === 0) {
    URL.revokeObjectURL(previewImg.src);
  }
  previewImg.removeAttribute('src');
}

btnAnalyze.addEventListener('click', async () => {
  if (!currentFile) return;
  stopOcclusionPolling();
  cleanupSession();
  occlusionB64 = null;
  segmentationB64 = null;
  currentOverlay = 'regions';
  setViewOption('segmentation', true);
  setViewOption('occlusion', true);
  setState('WAITING');
  btnAnalyze.disabled = true;
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
    showError('Error al analizar la imagen: ' + e.message);
  } finally {
    btnAnalyze.disabled = false;
  }
});

function renderResults(data) {
  const cls = data.prediction.class;
  const confidence = data.prediction.confidence;
  const diagEl = document.getElementById('diagnosis-class');
  const tbFlag = document.getElementById('tb-flag');
  const diagnosisNote = document.getElementById('diagnosis-note');

  diagEl.textContent = classLabels[cls] || cls;
  diagEl.className = 'diagnosis-class ' + cls;

  document.getElementById('confidence-value').textContent = (confidence * 100).toFixed(1) + '%';
  diagnosisNote.textContent = classNotes[cls] || '';

  tbFlag.textContent = data.prediction.tb_positive ? 'TB positivo' : 'TB negativo';
  tbFlag.className = 'tb-flag ' + (data.prediction.tb_positive ? 'positive' : 'negative');
  tbFlag.title = 'Umbral operativo de TB: ' + data.prediction.threshold.toFixed(3);

  const probabilities = data.probabilities;
  const health = probabilities.health;
  const sick = probabilities.sick;
  const tb = probabilities.tb;

  document.getElementById('bar-health').style.width = (health * 100) + '%';
  document.getElementById('bar-sick').style.width = (sick * 100) + '%';
  document.getElementById('bar-tb').style.width = (tb * 100) + '%';
  document.getElementById('val-health').textContent = (health * 100).toFixed(1) + '%';
  document.getElementById('val-sick').textContent = (sick * 100).toFixed(1) + '%';
  document.getElementById('val-tb').textContent = (tb * 100).toFixed(1) + '%';

  const topClass = ['health', 'sick', 'tb'].reduce((best, name) => (
    probabilities[name] > probabilities[best] ? name : best
  ), 'health');
  ['health', 'sick', 'tb'].forEach((name) => {
    document.getElementById('prob-' + name).classList.toggle('is-top', name === topClass);
  });

  const latency = data.metadata.inference_time_ms;
  document.getElementById('meta-threshold').textContent = data.prediction.threshold.toFixed(3);
  document.getElementById('meta-latency').textContent = latency >= 1000
    ? (latency / 1000).toFixed(1) + ' s'
    : Math.round(latency) + ' ms';
  document.getElementById('model-meta').textContent = 'La regresión logística combina textura y formas de la imagen para estimar la probabilidad.';

  baseImageB64 = previewImg.src || data.image_base64;
  segmentationB64 = data.segmentation_base64 || null;
  if (segmentationB64) setViewOption('segmentation', false);

  const regionsContainer = document.getElementById('regions-list');
  const featuresContainer = document.getElementById('features-list');
  regionsContainer.innerHTML = '';
  featuresContainer.innerHTML = '';

  if (data.level1 && Object.keys(data.level1.families).length) {
    const families = data.level1.families;
    for (const name of Object.keys(families)) {
      const value = families[name];
      regionsContainer.innerHTML += '<div class="region-item"><span class="region-name">' +
        escapeHtml(friendlyFamily(name)) + '</span><div class="region-bar"><div class="region-fill" style="width:' +
        (value * 100) + '%"></div></div><span class="region-val">' +
        (value * 100).toFixed(1) + '%</span></div>';
    }
  } else {
    regionsContainer.innerHTML = '<p class="empty-note">No hay información de influencia para esta imagen.</p>';
  }

  if (data.level1 && data.level1.top_features.length) {
    data.level1.top_features.forEach((feature, index) => {
      const increases = feature.direction === 'aumenta TB';
      const directionClass = increases ? 'up' : 'down';
      const sign = increases ? '+' : '−';
      featuresContainer.innerHTML += '<div class="feature-item">' +
        '<span class="fi-index">' + String(index + 1).padStart(2, '0') + '</span>' +
        '<span class="fi-name" title="' + escapeHtml(feature.name) + '">' + escapeHtml(friendlyFeature(feature.name)) + '</span>' +
        '<span class="fi-group">' + escapeHtml(friendlyGroup(feature.group)) + '</span>' +
        '<span class="fi-val ' + directionClass + '">' + sign + feature.importance.toFixed(3) + '</span>' +
        '</div>';
    });
  } else {
    featuresContainer.innerHTML = '<p class="empty-note">No hay características destacadas para mostrar.</p>';
  }

  regionImageB64 = baseImageB64;

  currentOverlay = 'regions';
  overlaySelect.value = 'regions';
  overlayLegend.hidden = true;
  drawOverlay('regions');
}

const viewTitles = {
  regions: 'Radiografía cargada',
  segmentation: 'Segmentación pulmonar',
  occlusion: 'Zonas que afectan la probabilidad de tuberculosis',
};

const viewNotes = {
  regions: 'La radiografía tal como la cargaste.',
  segmentation: 'Zona pulmonar detectada por el modelo. El contorno y el tinte marcan los pulmones.',
  occlusion: 'Zonas que más cambian la probabilidad estimada de tuberculosis al ocultarse.',
};

function drawOverlay(mode) {
  if (!baseImageB64) return;
  if (viewNote) viewNote.textContent = viewNotes[mode] || '';
  if (viewerTitle) viewerTitle.textContent = viewTitles[mode] || '';
  let src = baseImageB64;
  if (mode === 'regions' && regionImageB64) src = regionImageB64;
  else if (mode === 'segmentation' && segmentationB64) src = segmentationB64;
  else if (mode === 'occlusion' && occlusionB64) src = occlusionB64;

  const img = new Image();
  img.onload = () => {
    const isOcclusion = mode === 'occlusion';
    const natural = Math.max(img.naturalWidth, img.naturalHeight);
    const size = isOcclusion ? 600 : Math.min(1024, Math.max(natural, 512));
    imageCanvas.width = size;
    imageCanvas.height = size;
    canvasCtx.clearRect(0, 0, size, size);

    if (isOcclusion) {
      canvasCtx.drawImage(img, 0, 0, size, size);
      return;
    }

    canvasCtx.fillStyle = '#e3ddcf';
    canvasCtx.fillRect(0, 0, size, size);
    const scale = Math.min(size / img.naturalWidth, size / img.naturalHeight);
    const drawWidth = img.naturalWidth * scale;
    const drawHeight = img.naturalHeight * scale;
    canvasCtx.drawImage(
      img,
      (size - drawWidth) / 2,
      (size - drawHeight) / 2,
      drawWidth,
      drawHeight,
    );
  };
  img.src = src;
}

overlaySelect.addEventListener('change', () => {
  currentOverlay = overlaySelect.value;
  overlayLegend.hidden = currentOverlay !== 'occlusion' || !occlusionB64;
  drawOverlay(currentOverlay);
});

function stopOcclusionPolling() {
  if (occlusionInterval) {
    clearInterval(occlusionInterval);
    occlusionInterval = null;
  }
  occlusionStartedAt = null;
}

function startOcclusionPolling(sid) {
  stopOcclusionPolling();
  progressCard.hidden = false;
  progressFill.style.width = '0%';
  progressPercent.textContent = '0%';
  progressText.textContent = 'Calculando las zonas que más afectan la probabilidad de tuberculosis...';
  occlusionStartedAt = Date.now();

  fetch('/api/v1/predict/' + sid + '/occlude', { method: 'POST' }).catch(() => {});

  occlusionInterval = setInterval(async () => {
    try {
      const resp = await fetch('/api/v1/predict/' + sid + '/progress');
      const data = await resp.json();
      if (data.status === 'complete') {
        stopOcclusionPolling();
        if (!data.result || data.result.status === 'error') {
          progressPercent.textContent = '—';
          progressText.textContent = 'No se pudo calcular el mapa de zonas influyentes.';
          cleanupSession();
          return;
        }
        onOcclusionComplete(data.result);
      } else if (data.status === 'processing' && data.progress) {
        const progress = data.progress;
        const percent = Math.round((progress.current / progress.total) * 100);
        progressFill.style.width = percent + '%';
        progressPercent.textContent = percent + '%';
        progressText.textContent = 'Procesando región ' + progress.current + ' de ' + progress.total;
      } else if (data.status === 'not_started') {
        stopOcclusionPolling();
        progressPercent.textContent = '—';
        progressText.textContent = 'El mapa de zonas influyentes no está disponible para esta sesión.';
        cleanupSession();
        return;
      }

      if (occlusionStartedAt && Date.now() - occlusionStartedAt > OCCLUSION_TIMEOUT_MS) {
        stopOcclusionPolling();
        progressPercent.textContent = '—';
        progressText.textContent = 'El mapa de zonas influyentes tardó demasiado. Vuelve a analizar la imagen.';
        cleanupSession();
      }
    } catch (e) {
      // Reintentar en el siguiente ciclo.
    }
  }, 2000);
}

function onOcclusionComplete(res) {
  stopOcclusionPolling();
  progressFill.style.width = '100%';
  progressPercent.textContent = '100%';
  const elapsed = res && typeof res.time_ms === 'number'
    ? 'Mapa de zonas influyentes generado en ' + (res.time_ms / 1000).toFixed(1) + ' s'
    : 'Mapa de zonas influyentes generado.';
  progressText.textContent = elapsed;
  if (res && res.overlay) {
    occlusionB64 = res.overlay;
    setViewOption('occlusion', false);
    progressText.textContent = elapsed + ' Puedes verlo en el selector "Vista".';
  }
  setState('COMPLETE');
  cleanupSession();
}

function cleanupSession() {
  if (sessionId) {
    fetch('/api/v1/session/' + sessionId, { method: 'DELETE' }).catch(() => {});
  }
}

function setState(nextState) {
  state = nextState;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function showError(message) {
  stopOcclusionPolling();
  errorMessage.textContent = message;
  loadingSection.hidden = true;
  resultsSection.hidden = true;
  errorSection.hidden = false;
  setState('IDLE');
}




