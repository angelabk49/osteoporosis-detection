from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security.api_key import APIKeyHeader
from model import load_models, ensemble_predict, preprocess_image, assess_risk, clinical_adjust
from dataset import classes
import traceback
import os

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# ── Security ─────────────────────────────────────────────────────────────────
# API key loaded from environment variable (set in Render dashboard)
API_KEY = os.environ.get("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    """Validate API key if one is configured. If API_KEY env var is not set,
    auth is disabled (useful for local dev)."""
    if not API_KEY:
        return  # Auth disabled — no API_KEY configured
    if key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Include X-API-Key header."
        )

# ── App ───────────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

app = FastAPI(
    title="Osteoporosis Knee X-ray Classification API",
    description="Ensemble CNN classification of knee X-rays for osteoporosis risk assessment",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = load_models()

# Serve frontend static files (CSS, JS, images)
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    """Serve the frontend with the API key injected at runtime."""
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        with open(index, "r", encoding="utf-8") as f:
            html = f.read()
        # Inject the API key from env var into the placeholder in index.html
        html = html.replace(
            "window.API_KEY = '';",
            f"window.API_KEY = '{API_KEY}';"
        )
        return HTMLResponse(content=html)
    return {"status": "ok", "model": "ensemble (ResNet50 + DenseNet121 + EfficientNet-B0)"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "ensemble (ResNet50 + DenseNet121 + EfficientNet-B0)"}


@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(
    age: int = Form(...),
    sex: str = Form(...),
    vitamin_deficiency: bool = Form(...),
    xray: UploadFile = File(...)
):
    # ── 1. Validate file type ─────────────────────────────────────────────────
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if xray.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{xray.content_type}'. Upload a JPG or PNG image."
        )

    # ── 2. Validate file size (max 10 MB) ─────────────────────────────────────
    contents = await xray.read()
    file_size = len(contents)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size/1e6:.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── 3. Validate age ───────────────────────────────────────────────────────
    if not (1 <= age <= 120):
        raise HTTPException(status_code=400, detail="Age must be between 1 and 120.")

    # ── 4. Validate sex ───────────────────────────────────────────────────────
    if sex.lower() not in {"male", "female"}:
        raise HTTPException(status_code=400, detail="Sex must be 'Male' or 'Female'.")

    try:
        import io
        img = preprocess_image(io.BytesIO(contents))
        base_probs = ensemble_predict(models, img)

        # Adjust probabilities using clinical covariates (Bayesian prior)
        adjusted_probs = clinical_adjust(base_probs, age, sex, vitamin_deficiency)

        pred_idx = adjusted_probs.argmax(1).item()
        prediction = classes[pred_idx]
        confidence = adjusted_probs[0][pred_idx].item()

        class_probs = {cls: round(adjusted_probs[0][i].item(), 4) for i, cls in enumerate(classes)}
        urgency, message = assess_risk(prediction, age, sex, vitamin_deficiency)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probs,
            "urgency": urgency,
            "message": message,
            "patient": {
                "age": age,
                "sex": sex,
                "vitamin_deficiency": vitamin_deficiency
            }
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
