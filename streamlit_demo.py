from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


st.set_page_config(
    page_title="Indian Ethnic Clothing Classifier",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(201, 93, 44, 0.16), transparent 34%),
            radial-gradient(circle at top right, rgba(214, 168, 84, 0.18), transparent 28%),
            linear-gradient(180deg, #fffaf3 0%, #f7f1e7 100%);
        color: #2f241b;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(125, 40, 15, 0.95), rgba(191, 110, 42, 0.9));
        color: #fff7ef;
        box-shadow: 0 14px 32px rgba(76, 36, 14, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .hero p {
        margin: 0.55rem 0 0;
        font-size: 1rem;
        line-height: 1.5;
    }
    .card {
        background: rgba(255, 252, 246, 0.92);
        border: 1px solid rgba(150, 108, 74, 0.18);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 24px rgba(86, 60, 28, 0.08);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7e5b41;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #4c2b18;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
DATASET_DIR = BASE_DIR / "data" / "processed" / "dataset_subset" / "train"

DEFAULT_CLASS_NAMES: List[str] = sorted(
    [
        "blouse",
        "dhoti_pants",
        "dupattas",
        "gowns",
        "kurta_men",
        "leggings_and_salwars",
        "lehenga",
        "mojaris_men",
        "mojaris_women",
        "nehru_jackets",
        "palazzos",
        "petticoats",
        "saree",
        "sherwanis",
        "women_kurta",
    ]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

MODEL_OPTIONS: Dict[str, str] = {
    "MobileNetV2": "mobilenet_v2_best.pth",
    "DenseNet121": "densenet121_best.pth",
    "ResNet50": "resnet50_best.pth",
    "EfficientNetB0": "efficientnet_b0_best.pth",
}

MODEL_SUMMARY: Dict[str, Dict[str, str]] = {
    "MobileNetV2": {
        "accuracy": "77.42%",
        "macro_f1": "0.770",
        "notes": "Best overall result in this project.",
    },
    "DenseNet121": {
        "accuracy": "77.24%",
        "macro_f1": "0.767",
        "notes": "Very close to MobileNetV2.",
    },
    "ResNet50": {
        "accuracy": "73.96%",
        "macro_f1": "0.735",
        "notes": "A strong and reliable baseline.",
    },
    "EfficientNetB0": {
        "accuracy": "69.51%",
        "macro_f1": "0.685",
        "notes": "Works, but ranked below the others here.",
    },
}


def infer_class_names() -> List[str]:
    if DATASET_DIR.exists() and DATASET_DIR.is_dir():
        folder_names = sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()])
        if folder_names:
            return folder_names
    return DEFAULT_CLASS_NAMES


CLASS_NAMES = infer_class_names()
NUM_CLASSES = len(CLASS_NAMES)


def _make_classifier_head(in_features: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


def build_model(display_name: str, num_classes: int) -> nn.Module:
    if display_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        head = _make_classifier_head(in_features, num_classes)
        model.classifier = nn.Sequential(model.classifier[0], head)
        return model

    if display_name == "DenseNet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = _make_classifier_head(in_features, num_classes)
        return model

    if display_name == "ResNet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = _make_classifier_head(in_features, num_classes)
        return model

    if display_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        head = _make_classifier_head(in_features, num_classes)
        model.classifier = nn.Sequential(model.classifier[0], head)
        return model

    raise ValueError(f"Unsupported model: {display_name}")


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format.")


@st.cache_resource(show_spinner=False)
def load_model(display_name: str) -> nn.Module:
    checkpoint_name = MODEL_OPTIONS[display_name]
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {checkpoint_name}. Please place it in outputs/checkpoints/."
        )

    model = build_model(display_name, NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = extract_state_dict(checkpoint)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys or unexpected_keys:
        raise RuntimeError("This saved model could not be loaded for the selected option.")

    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model: nn.Module, image: Image.Image, top_k: int = 3) -> Tuple[str, List[Tuple[str, float]]]:
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(CLASS_NAMES)), dim=1)

    top_probs = top_probs.squeeze(0).cpu().tolist()
    top_indices = top_indices.squeeze(0).cpu().tolist()

    predictions = [(CLASS_NAMES[idx], float(prob) * 100.0) for idx, prob in zip(top_indices, top_probs)]
    return predictions[0][0], predictions


st.markdown(
    """
    <div class="hero">
        <h1>Indian Ethnic Clothing Classifier</h1>
        <p>Try a trained model by uploading a clothing image. The app shows the most likely class and the top matching predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Choose a model")
    selected_model = st.selectbox("Available models", list(MODEL_OPTIONS.keys()), index=0)
    st.markdown("---")
    st.subheader("Model snapshot")
    st.markdown(
        f'<div class="metric-label">Accuracy</div><div class="metric-value">{MODEL_SUMMARY[selected_model]["accuracy"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-label">Macro F1</div><div class="metric-value">{MODEL_SUMMARY[selected_model]["macro_f1"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption(MODEL_SUMMARY[selected_model]["notes"])

intro_col, info_col = st.columns([1.35, 1])

with intro_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload an image")
    uploaded_file = st.file_uploader(
        "Choose a clothing photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear image with one main clothing item for the best result.",
    )
    st.caption("Supported formats: JPG, JPEG, PNG, WEBP")
    st.markdown('</div>', unsafe_allow_html=True)

with info_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About this demo")
    st.write(f"Running on: {DEVICE}")
    st.write(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    st.write(f"Available models: {len(MODEL_OPTIONS)}")
    st.caption("Classes supported by the app")
    st.write(", ".join(CLASS_NAMES))
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Upload an image to start the prediction.")
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        preview_col, result_col = st.columns([1.05, 1])
        with preview_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Your image")
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("Checking the image..."):
            model = load_model(selected_model)
            predicted_label, predictions = predict_image(model, image, top_k=3)

        with result_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Prediction result")
            st.success(f"Predicted class: {predicted_label}")
            st.markdown("### Top matches")
            for label, score in predictions:
                st.write(f"{label}: {score:.2f}%")
                st.progress(max(0.0, min(score / 100.0, 1.0)))
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as exc:
        st.error(str(exc))

st.markdown("---")
st.subheader("How to use this app")
st.markdown(
    """
- Upload a clear clothing image and choose a model from the sidebar.
    """
)
