import torch
import torch.nn.functional as F
import numpy as np
from dataset import get_resnet, get_densenet, get_effnet, classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tta_transforms = [
    lambda x: x,
    lambda x: torch.flip(x, [-1]),
    lambda x: torch.flip(x, [-2]),
    lambda x: torch.rot90(x, 1, [-2, -1]),
    lambda x: torch.rot90(x, 3, [-2, -1])
]


def load_models():
    resnet = get_resnet()
    densenet = get_densenet()
    effnet = get_effnet()

    resnet.load_state_dict(torch.load("checkpoints/resnet_best.pth", map_location=device))
    densenet.load_state_dict(torch.load("checkpoints/densenet_best.pth", map_location=device))
    effnet.load_state_dict(torch.load("checkpoints/effnet_best.pth", map_location=device))

    for m in [resnet, densenet, effnet]:
        m.eval()

    return [resnet, densenet, effnet]


def tta_predict(model, x):
    probs = torch.zeros((1, len(classes))).to(device)
    with torch.no_grad():
        for tf in tta_transforms:
            probs += F.softmax(model(tf(x)), dim=1)
    return probs / len(tta_transforms)


# Ensemble weights: [ResNet-50, DenseNet-121, EfficientNet-B0]
ENSEMBLE_WEIGHTS = [0.40, 0.35, 0.25]


def ensemble_predict(models, x):
    weighted_probs = sum(
        w * tta_predict(m, x)
        for m, w in zip(models, ENSEMBLE_WEIGHTS)
    )
    return weighted_probs


def clinical_adjust(probs_tensor, age, sex, vitamin_def):
    """
    Adjust CNN ensemble probabilities using clinical covariates.

    Uses evidence-based risk multipliers derived from osteoporosis
    epidemiological literature to shift predictions based on patient
    demographics, acting as a Bayesian prior on the CNN output.

    Args:
        probs_tensor: (1, 3) tensor — [normal, osteopenia, osteoporosis]
        age: int — patient age
        sex: str — "male" or "female"
        vitamin_def: bool — history of vitamin deficiency

    Returns:
        Adjusted (1, 3) tensor with renormalized probabilities
    """
    probs = probs_tensor.detach().cpu().numpy().flatten()
    risk_mult = np.ones(3)  # [normal, osteopenia, osteoporosis]

    # Age-based risk adjustment
    if age >= 70:
        risk_mult[0] *= 0.70
        risk_mult[1] *= 1.25
        risk_mult[2] *= 1.50
    elif age >= 60:
        risk_mult[0] *= 0.80
        risk_mult[1] *= 1.15
        risk_mult[2] *= 1.30
    elif age >= 50:
        risk_mult[0] *= 0.90
        risk_mult[1] *= 1.10
        risk_mult[2] *= 1.15

    # Sex-based risk (females at higher risk post-menopause)
    if sex.lower() == "female":
        if age >= 65:
            risk_mult[0] *= 0.75
            risk_mult[1] *= 1.20
            risk_mult[2] *= 1.40
        elif age >= 50:
            risk_mult[0] *= 0.85
            risk_mult[1] *= 1.15
            risk_mult[2] *= 1.20

    # Vitamin deficiency history
    if vitamin_def:
        risk_mult[0] *= 0.80
        risk_mult[1] *= 1.15
        risk_mult[2] *= 1.35

    # Apply multipliers and renormalize
    adjusted = probs * risk_mult
    adjusted = adjusted / adjusted.sum()

    return torch.tensor(adjusted, dtype=torch.float32).unsqueeze(0).to(probs_tensor.device)


def assess_risk(prediction, age, sex, vitamin_def):
    """Determine clinical urgency based on prediction and patient profile."""
    urgency = "Low"
    message = "No immediate concern."

    if vitamin_def:
        urgency = "Moderate"
        message = "Increased risk due to vitamin deficiency."

    if vitamin_def and age >= 60:
        urgency = "High"
        message = "High risk: age and vitamin deficiency."

    if sex.lower() == "female" and age >= 65:
        urgency = "High"
        message = "High risk: post-menopausal age group."

    if prediction == "osteoporosis":
        urgency = "Critical"
        message = "Immediate medical consultation recommended."

    return urgency, message


# ─── Preprocessing ───────────────────────────────────────────────────────────
from torchvision import transforms
from PIL import Image
from dataset import CLAHETransform

transform = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = transform(img).unsqueeze(0)
    return img
