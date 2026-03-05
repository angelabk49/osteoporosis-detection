/**
 * OsteoScan AI — Frontend Logic
 * Handles: drag-and-drop upload, form submission, API call, results rendering
 */

// ── API URL ───────────────────────────────────────────────────────────────
// In production (HF Spaces / Render): frontend and API are on the same server,
// so we use a relative URL. In local dev, point to localhost:8000.
const API_URL = (location.hostname === '127.0.0.1' || location.hostname === 'localhost')
  ? 'http://127.0.0.1:8000'
  : '';   // empty = same origin (relative URLs)

// ===== DOM References =====
const navbar = document.getElementById('navbar');
const form = document.getElementById('analysis-form');
const btnAnalyze = document.getElementById('btn-analyze');
const uploadZone = document.getElementById('upload-zone');
const uploadInput = document.getElementById('upload-input');
const uploadPreview = document.getElementById('upload-preview');
const fileInfo = document.getElementById('file-info');
const resultsPlaceholder = document.getElementById('results-placeholder');
const resultsContent = document.getElementById('results-content');
const toast = document.getElementById('toast');

let selectedFile = null;

// ===== Navbar scroll effect =====
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
});

// ===== Intersection Observer for fade-in animations =====
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
);

document.querySelectorAll('.fade-in').forEach((el) => observer.observe(el));

// ===== Upload Zone — Click =====
uploadZone.addEventListener('click', () => uploadInput.click());

// ===== Upload Zone — Drag & Drop =====
['dragenter', 'dragover'].forEach((evt) => {
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((evt) => {
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
  });
});

uploadZone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// ===== Upload Zone — File Input Change =====
uploadInput.addEventListener('change', () => {
  if (uploadInput.files[0]) handleFile(uploadInput.files[0]);
});

function handleFile(file) {
  const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
  if (!validTypes.includes(file.type)) {
    showToast('Please upload a JPG or PNG image.');
    return;
  }

  selectedFile = file;
  uploadZone.classList.add('has-file');

  // Image preview
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadPreview.src = e.target.result;
  };
  reader.readAsDataURL(file);

  // File info text
  const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
  fileInfo.textContent = `${file.name} · ${sizeMB} MB`;
}

// ===== Form Submission =====
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const age = document.getElementById('input-age').value;
  const sex = document.getElementById('input-sex').value;
  const vitaminDef = document.getElementById('input-vitamin').checked;

  // Validation
  if (!age || age < 1 || age > 120) {
    showToast('Please enter a valid age (1–120).');
    return;
  }
  if (!sex) {
    showToast('Please select patient sex.');
    return;
  }
  if (!selectedFile) {
    showToast('Please upload a knee X-ray image.');
    return;
  }

  // Build FormData
  const formData = new FormData();
  formData.append('age', parseInt(age));
  formData.append('sex', sex);
  formData.append('vitamin_deficiency', vitaminDef);
  formData.append('xray', selectedFile);

  // UI → loading state
  btnAnalyze.classList.add('loading');
  btnAnalyze.disabled = true;

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${response.status})`);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    console.error('Prediction error:', err);
    showToast(err.message || 'Failed to connect to the server. Make sure the backend is running.');
  } finally {
    btnAnalyze.classList.remove('loading');
    btnAnalyze.disabled = false;
  }
});

// ===== Render Results =====
function renderResults(data) {
  // Hide placeholder, show content
  resultsPlaceholder.style.display = 'none';
  resultsContent.classList.add('active');

  // Scroll to results on mobile
  resultsContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Prediction
  const predEl = document.getElementById('result-prediction');
  predEl.textContent = capitalize(data.prediction);
  predEl.className = 'prediction-value ' + data.prediction.toLowerCase();

  // Confidence bar
  const confPercent = (data.confidence * 100).toFixed(1);
  document.getElementById('result-confidence').textContent = confPercent + '%';
  // Use requestAnimationFrame to trigger the CSS transition
  requestAnimationFrame(() => {
    document.getElementById('confidence-bar').style.width = confPercent + '%';
  });

  // Class probabilities
  if (data.class_probabilities) {
    for (const [cls, prob] of Object.entries(data.class_probabilities)) {
      const pct = (prob * 100).toFixed(1);
      const barEl = document.getElementById('prob-' + cls);
      const valEl = document.getElementById('prob-' + cls + '-val');
      if (barEl) {
        requestAnimationFrame(() => {
          barEl.style.width = pct + '%';
        });
      }
      if (valEl) valEl.textContent = pct + '%';
    }
  }

  // Urgency
  const urgencyKey = data.urgency.toLowerCase();
  const urgencyBlock = document.getElementById('urgency-block');
  urgencyBlock.className = 'urgency-block ' + urgencyKey;

  const urgencyIcons = {
    low: '✅',
    moderate: '⚠️',
    high: '🔶',
    critical: '🚨',
  };

  document.getElementById('urgency-icon').textContent = urgencyIcons[urgencyKey] || '❓';

  const urgencyLevelEl = document.getElementById('urgency-level');
  urgencyLevelEl.textContent = data.urgency + ' Risk';
  urgencyLevelEl.className = 'urgency-level ' + urgencyKey;

  document.getElementById('urgency-message').textContent = data.message;
}

// ===== Toast Notification =====
let toastTimeout = null;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('visible');

  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.classList.remove('visible');
  }, 4000);
}

// ===== Utility =====
function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}
