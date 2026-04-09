const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('file-input');
const previewImg   = document.getElementById('preview-img');
const dropText     = document.getElementById('drop-text');
const browseLink   = document.getElementById('browse-link');
const analyzeBtn   = document.getElementById('analyze-btn');
const errorBox     = document.getElementById('error-box');
const tabImage     = document.getElementById('tab-image');
const tabVideo     = document.getElementById('tab-video');
const panelImage   = document.getElementById('panel-image');
const panelVideo   = document.getElementById('panel-video');
const gaugeOverlay = document.getElementById('gauge-overlay');
const gaugeFill    = document.getElementById('gauge-fill');
const gaugePct     = document.getElementById('gauge-pct');
const gaugeLbl     = document.getElementById('gauge-lbl');
const verdictBadge = document.getElementById('verdict-badge');
const imgMeta      = document.getElementById('img-meta');
const clearBtn     = document.getElementById('clear-btn');

// Arc: cx=130 cy=140 r=116 → circumference fraction for 240° sweep
const GAUGE_TOTAL = (240 / 360) * 2 * Math.PI * 116; // ≈ 485

let currentFile = null;

/* ── Tabs ──────────────────────────────────────────────────────── */
tabImage.addEventListener('click', () => {
    tabImage.classList.add('active'); tabVideo.classList.remove('active');
    panelImage.classList.remove('hidden'); panelVideo.classList.add('hidden');
});
tabVideo.addEventListener('click', () => {
    tabVideo.classList.add('active'); tabImage.classList.remove('active');
    panelVideo.classList.remove('hidden'); panelImage.classList.add('hidden');
});

/* ── Helpers ───────────────────────────────────────────────────── */
function showError(msg) { errorBox.textContent = '⚠ ' + msg; errorBox.style.display = 'block'; }
function hideError()    { errorBox.style.display = 'none'; }

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function resetGauge() {
    gaugeOverlay.classList.remove('visible');
    verdictBadge.className   = 'verdict-badge';
    verdictBadge.textContent = '';
    gaugeFill.style.strokeDasharray = '0 999';
    gaugePct.textContent = '0%';
    gaugeLbl.textContent = 'CONFIDENCE';
    gaugePct.style.color = '#1a1f36';
    gaugeLbl.style.color = '#6b7a99';
}

function clearImage() {
    currentFile = null;
    previewImg.src = '#';
    previewImg.style.display = 'none';
    previewImg.classList.remove('loaded');
    dropText.style.display = 'flex';
    dropZone.classList.remove('has-image');
    if (imgMeta) imgMeta.style.display = 'none';
    if (clearBtn) clearBtn.style.display = 'none';
    analyzeBtn.style.display = 'none';
    fileInput.value = '';
    resetGauge();
    hideError();
}

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.style.display = 'block';
        previewImg.classList.remove('loaded');
        dropText.style.display = 'none';
        dropZone.classList.add('has-image');

        previewImg.onload = () => {
            previewImg.classList.add('loaded');
            // Show metadata
            if (imgMeta) {
                imgMeta.textContent = `${previewImg.naturalWidth} × ${previewImg.naturalHeight}px  ·  ${formatBytes(file.size)}  ·  ${file.name}`;
                imgMeta.style.display = 'block';
            }
            if (clearBtn) clearBtn.style.display = 'flex';
            analyzeBtn.style.display = 'block';
        };
        previewImg.src = e.target.result;
        resetGauge();
    };
    reader.readAsDataURL(file);
}

/* ── File / drag-drop ──────────────────────────────────────────── */
browseLink.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener('click', () => { if (!currentFile) fileInput.click(); });

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) { currentFile = fileInput.files[0]; hideError(); showPreview(currentFile); }
});

['dragover','dragleave','drop'].forEach(n =>
    dropZone.addEventListener(n, e => { e.preventDefault(); e.stopPropagation(); })
);
dropZone.addEventListener('dragover',  () => dropZone.classList.add('drag-over'));
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
    dropZone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) { currentFile = f; hideError(); showPreview(f); }
});

if (clearBtn) clearBtn.addEventListener('click', e => { e.stopPropagation(); clearImage(); });

/* ── Paste support ─────────────────────────────────────────────── */
document.addEventListener('paste', e => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
        if (item.type.startsWith('image/')) {
            const f = item.getAsFile();
            if (f) { currentFile = f; hideError(); showPreview(f); }
            break;
        }
    }
});

/* ── Analyze ────────────────────────────────────────────────────── */
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) { showError('Please select an image first.'); return; }
    hideError();
    analyzeBtn.disabled  = true;
    analyzeBtn.innerHTML = '<span class="spinner"></span>Analyzing…';

    try {
        const fd = new FormData();
        fd.append('file', currentFile);
        const res  = await fetch('/analyze', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.error) showError(data.error);
        else renderResult(data.prediction, data.confidence);
    } catch {
        showError('Network error. Please try again.');
    } finally {
        analyzeBtn.disabled  = false;
        analyzeBtn.innerHTML = 'Analyze Image';
    }
});

/* ── Render result ─────────────────────────────────────────────── */
function renderResult(prediction, confidence) {
    const isReal = prediction === 'Real';
    const color  = isReal ? '#00b87a' : '#e8365d';
    const gradId = isReal ? 'url(#g-real)' : 'url(#g-fake)';

    gaugePct.style.color = color;
    gaugeLbl.style.color = color;
    gaugeLbl.textContent = isReal ? 'AUTHENTIC' : 'SYNTHETIC';
    gaugeFill.setAttribute('stroke', gradId);

    // Animate arc
    const drawn = GAUGE_TOTAL * (confidence / 100);
    gaugeFill.style.strokeDasharray = '0 999';
    requestAnimationFrame(() => requestAnimationFrame(() => {
        gaugeFill.style.strokeDasharray = `${drawn} ${GAUGE_TOTAL + 20}`;
    }));

    // Animate counter
    animateCounter(0, confidence, 1200, n => { gaugePct.textContent = n + '%'; });

    // Verdict badge
    verdictBadge.textContent = isReal ? '✔  Real Image' : '⚠  Deepfake Detected';
    verdictBadge.className   = 'verdict-badge ' + (isReal ? 'real' : 'fake');

    // Show overlay
    gaugeOverlay.classList.add('visible');
}

function animateCounter(from, to, ms, cb) {
    const t0 = performance.now();
    (function tick(now) {
        const p = Math.min((now - t0) / ms, 1);
        cb(Math.round(from + (to - from) * (1 - Math.pow(1 - p, 4))));
        if (p < 1) requestAnimationFrame(tick);
    })(t0);
}
