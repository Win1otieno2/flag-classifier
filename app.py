
import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from pathlib import Path

# --- CONFIG ---
MODEL_PATH = Path("flag_classifier_resnet50.pth")
CLASS_FILE = Path("class_names.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# --- LOAD CLASS NAMES ---
with open(CLASS_FILE, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines() if line.strip()]

# --- MODEL ---
model = models.resnet50(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, len(classes))
)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- PREDICTION FUNCTION ---
def predict_flag(image):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top5_prob, top5_idx = torch.topk(probs, 5)
    
    results = [
        (classes[i], float(top5_prob[j]))
        for j, i in enumerate(top5_idx)
    ]
    return {name: prob for name, prob in results}

# --- GRADIO APP ---
title = "🏳️ Flag Classifier (ResNet-50)"
description = """
Upload a flag image and the model will predict which country's flag it is.
Trained on over 200 world flags using a fine-tuned ResNet-50.
"""

interface = gr.Interface(
    fn=predict_flag,
    inputs=gr.Image(type="pil", label="Upload Flag"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    title=title,
    description=description,
    examples=[
        ["examples/kenya.png"],
        ["examples/united_states.png"],
        ["examples/japan.png"],
    ] if Path("examples").exists() else None,
)

if __name__ == "__main__":
    interface.launch()
