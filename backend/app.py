from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from model import load_models, ensemble_predict, preprocess_image, assess_risk, clinical_adjust
from dataset import classes
import traceback
import os

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = FastAPI(
    title="Osteoporosis Knee X-ray Classification API",
    description="Ensemble CNN classification of knee X-rays for osteoporosis risk assessment",
    version="1.0.0"
)

# CORS — allow the frontend served from any local origin
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
    """Serve the frontend web app."""
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "ok", "model": "ensemble (ResNet50 + DenseNet121 + EfficientNet-B0)"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "ensemble (ResNet50 + DenseNet121 + EfficientNet-B0)"}


@app.post("/predict")
async def predict(
    age: int = Form(...),
    sex: str = Form(...),
    vitamin_deficiency: bool = Form(...),
    xray: UploadFile = File(...)
):
    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if xray.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{xray.content_type}'. Upload a JPG or PNG image."
        )

    try:
        img = preprocess_image(xray.file)
        base_probs = ensemble_predict(models, img)

        # Adjust probabilities using clinical covariates (Bayesian prior)
        adjusted_probs = clinical_adjust(base_probs, age, sex, vitamin_deficiency)

        pred_idx = adjusted_probs.argmax(1).item()
        prediction = classes[pred_idx]
        confidence = adjusted_probs[0][pred_idx].item()

        # Get per-class probabilities for the frontend
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
