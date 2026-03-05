# 🦴 ODS — Osteoporosis Detection System

An AI-powered web application for early osteoporosis detection from knee X-ray images. Uses an ensemble of three deep learning models with clinical covariate integration to provide risk assessments.

---

## ✨ Features

- **Ensemble CNN Classification** — Combines ResNet-50, DenseNet-121, and EfficientNet-B0 for robust predictions
- **Clinical Covariate Integration** — Patient age, sex, and vitamin deficiency history directly influence prediction probabilities via Bayesian prior adjustment
- **CLAHE Preprocessing** — Contrast-Limited Adaptive Histogram Equalization enhances bone density visibility in X-rays
- **Test-Time Augmentation (TTA)** — Evaluates each image from 5 orientations for more reliable predictions
- **Risk Assessment** — Combines AI prediction with patient demographics for a clinical urgency rating
- **Modern Web Interface** — Responsive, dark-themed UI with drag-and-drop upload and animated results

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (HTML/CSS/JS)            │
│  Upload X-ray → Patient Details → API Call          │
└───────────────────────┬─────────────────────────────┘
                        │ POST /predict
                        ▼
┌─────────────────────────────────────────────────────┐
│                 FastAPI Backend                       │
│                                                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ ResNet-50│  │ DenseNet-121 │  │EfficientNet-B0│  │
│  │  (40%)   │  │    (35%)     │  │    (25%)      │  │
│  └────┬─────┘  └──────┬───────┘  └──────┬────────┘  │
│       └───────────┬────┘                 │           │
│              Weighted Ensemble           │           │
│                   ┌──────────────────────┘           │
│                   ▼                                  │
│          Clinical Adjustment                         │
│    (age, sex, vitamin deficiency)                    │
│                   ▼                                  │
│          Risk Assessment                             │
│    (Low / Moderate / High / Critical)                │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **Deep Learning** | PyTorch, TorchVision |
| **Models** | ResNet-50, DenseNet-121, EfficientNet-B0 |
| **Image Processing** | OpenCV (CLAHE), Pillow |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Deployment** | Docker |

---

## 📁 Project Structure

```
Osteoporosis Knee X ray/
├── .gitignore
├── README.md
├── backend/
│   ├── Dockerfile              # Docker deployment config
│   ├── requirements.txt        # Python dependencies
│   ├── app.py                  # FastAPI server & endpoints
│   ├── model.py                # Inference: ensemble, TTA, clinical adjustment
│   ├── dataset.py              # Data loading, transforms, model definitions
│   ├── train.py                # Model training with early stopping
│   ├── finetune.py             # Fine-tuning on held-out data
│   ├── test.py                 # Evaluation & metrics (ROC, confusion matrix)
│   ├── diagnose.py             # Detailed diagnostic report
│   ├── weight_search.py        # Grid search for optimal ensemble weights
│   ├── cross_val.py            # K-fold cross-validation
│   ├── augment_dataset.py      # Offline data augmentation
│   ├── api_test.py             # API endpoint test suite
│   ├── quick_test.py           # Quick API smoke test
│   ├── efficientnet_b0_local.pth  # Pretrained EfficientNet weights
│   └── checkpoints/
│       ├── resnet_best.pth     # Trained ResNet-50 weights
│       ├── densenet_best.pth   # Trained DenseNet-121 weights
│       └── effnet_best.pth     # Trained EfficientNet-B0 weights
├── backend_original/           # Backup of original code (with AlexNet)
├── data/
│   ├── train/                  # Training images (normal/osteopenia/osteoporosis)
│   └── test/                   # Test images (normal/osteopenia/osteoporosis)
└── frontend/
    ├── index.html              # Main web page
    ├── styles.css              # Styling (dark theme, animations)
    ├── script.js               # Frontend logic (upload, API call, rendering)
    └── FOR TESTING/            # Held-out test images
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (3.11 recommended)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/osteoporosis-detection.git
cd "Osteoporosis Knee X ray"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### Train the Models

```bash
cd backend

# Train all three models (ResNet-50, DenseNet-121, EfficientNet-B0)
python train.py

# (Optional) Fine-tune on held-out data
python finetune.py

# (Optional) Find optimal ensemble weights
python weight_search.py

# (Optional) Evaluate on test set
python test.py
```

### Run the Application

**Terminal 1 — Start the Backend:**
```bash
cd backend
uvicorn app:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Serve the Frontend:**
```bash
cd frontend
python -m http.server 5500
```

Then open [http://127.0.0.1:5500](http://127.0.0.1:5500) in your browser.

---

## 📡 API Reference

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "ensemble (ResNet50 + DenseNet121 + EfficientNet-B0)"
}
```

### `POST /predict`
Classify a knee X-ray image.

**Request (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `xray` | File | ✅ | Knee X-ray image (JPG/PNG) |
| `age` | int | ✅ | Patient age (1–120) |
| `sex` | string | ✅ | "Male" or "Female" |
| `vitamin_deficiency` | bool | ✅ | History of vitamin deficiency |

**Response:**
```json
{
  "prediction": "osteopenia",
  "confidence": 0.6234,
  "class_probabilities": {
    "normal": 0.1823,
    "osteopenia": 0.6234,
    "osteoporosis": 0.1943
  },
  "urgency": "High",
  "message": "High risk: post-menopausal age group.",
  "patient": {
    "age": 65,
    "sex": "Female",
    "vitamin_deficiency": true
  }
}
```

---

## 🧠 Model Details

### Ensemble Architecture

| Model | Weight | Input | Parameters |
|-------|--------|-------|------------|
| ResNet-50 | 40% | 384×384 | 25.6M |
| DenseNet-121 | 35% | 384×384 | 8.0M |
| EfficientNet-B0 | 25% | 384×384 | 5.3M |

### Preprocessing Pipeline

1. **CLAHE** — Enhances bone density contrast in grayscale X-rays
2. **Resize** — 384×384 pixels (higher resolution preserves trabecular patterns)
3. **Normalize** — ImageNet mean/std normalization

### Clinical Covariate Adjustment

The system adjusts CNN output probabilities using a Bayesian prior based on:

| Factor | Effect on Osteoporosis Probability |
|--------|------------------------------------|
| Age ≥ 70 | ↑ 1.50× |
| Female ≥ 65 | ↑ 1.40× |
| Vitamin Deficiency | ↑ 1.35× |

### Risk Assessment Logic

| Urgency | Condition |
|---------|-----------|
| **Low** | No risk factors detected |
| **Moderate** | Vitamin deficiency present |
| **High** | Age ≥ 60 + vitamin deficiency, OR female ≥ 65 |
| **Critical** | AI predicts osteoporosis |

### Data Augmentations (Training)

- Random horizontal/vertical flip
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation)
- Random affine (translate, scale)
- Random perspective distortion
- Gaussian blur
- Random erasing

### Test-Time Augmentation (Inference)

Each image is evaluated with 5 transforms: identity, horizontal flip, vertical flip, 90° rotation, 270° rotation. Predictions are averaged for robustness.

---

## 🐳 Docker Deployment

```bash
cd backend

# Build
docker build -t osteo-api .

# Run
docker run -p 8000:8000 osteo-api
```

---

## 🌐 Deployment Options

### Option A: Render (Recommended)
1. Push code to GitHub (use Git LFS for `.pth` files)
2. Create a **Web Service** on [render.com](https://render.com)
3. Set root directory to `backend`, runtime to Docker
4. Deploy frontend as a **Static Site** (publish directory: `frontend`)

### Option B: Railway
1. Connect GitHub repo at [railway.app](https://railway.app)
2. Set root directory to `backend`
3. Railway auto-detects the Dockerfile

### Option C: Hugging Face Spaces
1. Create a new Space with Docker SDK
2. Upload backend code + model weights
3. Ideal for ML demos with free GPU access

---

## 🧪 Testing

```bash
# Start the backend first, then run:
cd backend

# Full API test suite (12 test cases, generates HTML report)
python api_test.py

# Quick smoke test (3 images)
python quick_test.py

# Model evaluation with metrics
python test.py

# Detailed diagnostic report
python diagnose.py
```

---

## ⚠️ Disclaimer

This tool is an **AI-assisted screening aid** and is not a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for diagnosis and treatment decisions. AI predictions can be incorrect — clinical verification is essential.

---

## 📜 License

This project is for educational and research purposes.
