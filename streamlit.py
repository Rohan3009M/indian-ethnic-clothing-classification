import json
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


st.set_page_config(
    page_title="Indian Ethnic Clothing Classifier",
    page_icon="👗",
    layout="wide",
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
        "macro_f1": "0.769",
        "notes": "Best overall performer on the evaluation subset.",
    },
    "DenseNet121": {
        "accuracy": "77.24%",
        "macro_f1": "0.766",
        "notes": "Very close to MobileNetV2 in overall performance.",
    },
    "ResNet50": {
        "accuracy": "73.96%",
        "macro_f1": "0.735",
        "notes": "Good baseline transfer-learning model.",
    },
    "EfficientNetB0": {
        "accuracy": "69.51%",
        "macro_f1": "0.685",
        "notes": "Lower performance than the others on this setup.",
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
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(display_name, NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = extract_state_dict(checkpoint)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint and model architecture do not match. "
            f"Missing keys: {missing_keys[:5]} | Unexpected keys: {unexpected_keys[:5]}"
        )

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


st.title("👗 Indian Ethnic Clothing Classification Demo")
st.caption("Upload an image, select a trained model, and compare predictions.")

with st.sidebar:
    st.header("Model Selection")
    selected_model = st.selectbox("Choose a model", list(MODEL_OPTIONS.keys()), index=0)
    st.markdown("---")
    st.subheader("Model Performance")
    st.write(f"**Accuracy:** {MODEL_SUMMARY[selected_model]['accuracy']}")
    st.write(f"**Macro F1:** {MODEL_SUMMARY[selected_model]['macro_f1']}")
    st.caption(MODEL_SUMMARY[selected_model]["notes"])

left_col, right_col = st.columns([1.1, 1])

with left_col:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    st.markdown("### Supported classes")
    st.write(", ".join(CLASS_NAMES))

with right_col:
    st.markdown("### App details")
    st.write(f"**Device:** {DEVICE}")
    st.write(f"**Input size:** {IMAGE_SIZE} × {IMAGE_SIZE}")
    st.write(f"**Models available:** {len(MODEL_OPTIONS)}")

if uploaded_file is None:
    st.info("Upload an image to test the model.")
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        preview_col, result_col = st.columns([1, 1])
        with preview_col:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running prediction..."):
            model = load_model(selected_model)
            predicted_label, predictions = predict_image(model, image, top_k=3)

        with result_col:
            st.subheader("Prediction")
            st.success(f"Predicted class: {predicted_label}")
            st.subheader("Top probabilities")
            for label, score in predictions:
                st.write(f"**{label}** — {score:.2f}%")
                st.progress(max(0.0, min(score / 100.0, 1.0)))

    except Exception as exc:
        st.error(str(exc))

st.markdown("---")
st.markdown("### Notes for use")
st.markdown(
    "- Keep model checkpoint files inside `outputs/checkpoints/`.\n"
    "- Use clothing images similar to the training dataset for better predictions.\n"
    "- If a checkpoint fails to load, check whether the saved architecture matches the selected model."
)
