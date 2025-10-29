# 🏳️ Flag Classification with Deep Learning

## 🌍 Overview
This project teaches a computer to recognize **country flags** using **deep learning**.
I trained a model so that when you show it an image of a flag, it predicts which country it belongs to. It trains a **ResNet-50 deep learning model** to recognize the national flags of the world.  
Using **PyTorch**, **torchvision**, and **Gradio**, the model learns to classify over **200 countries and territories** based on their flags — achieving over **94% accuracy** on the test set.

---

## 🚀 Features
- 🧠 **Transfer Learning** with ResNet-50 for high accuracy  
- 🧩 **Image Augmentation** (rotation, brightness, perspective, etc.)  
- 📊 **Evaluation Tools** — confusion matrix, classification report, top-5 accuracy  
- 💬 **Interactive App** via Gradio or Hugging Face Spaces  
- 🏗️ **Clean Dataset Structure** with train/test split  
- 📁 Automatically reads class names from `class_names.txt`

---

## 🧰 Project Structure
```
flags/
│
├── app.py                     # Gradio web app
├── train_resnet50.py          # Training script with augmentations
├── confusion_matrix.py        # Evaluation and diagnostics
├── class_names.txt            # List of country names (one per line)
├── flag_classifier_resnet50.pth  # Trained model weights
│
├── flags_dataset/
│   ├── train/
│   │   ├── Kenya/
│   │   ├── United_States/
│   │   └── ...
│   └── test/
│       ├── Kenya/
│       ├── United_States/
│       └── ...
│
└── examples/                  # Optional sample images for app demo
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/win1otieno2/flag-classifier.git
cd flag-classifier

# Create environment
python -m venv .venv
source .venv/bin/activate      # (on Windows: .venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` should include:
```
torch
torchvision
gradio
pillow
matplotlib
scikit-learn
```

---

## 🧠 Training
```bash
python train_resnet50.py
```

- Uses pretrained **ResNet-50**
- Applies color jitter, rotation, and perspective transforms  
- Trains for 20 epochs by default  
- Saves model as `flag_classifier_resnet50.pth`

---

## 🔍 Evaluation
```bash
python confusion_matrix.py
```
Generates:
- ✅ Overall accuracy
- 📊 Classification report
- 🧩 Confusion matrix (`confusion_matrix.png`)
- ⚠️ Top-5 most confused flag pairs (e.g., Liberia ↔ USA)

---

## 🌐 Interactive Demo
```bash
python app.py
```

Runs a **Gradio web app** where you can:
- Upload any flag image (`.png`, `.jpg`)
- See top-5 predicted countries with confidence scores

Deploy easily on **Hugging Face Spaces** — just include:
```
app.py
flag_classifier_resnet50.pth
class_names.txt
requirements.txt
```

---

## 📈 Results

| Metric | Value |
|:--------|:------:|
| Training Accuracy | 95.0% |
| Test Accuracy | 93.6% |
| Model | ResNet-50 (fine-tuned) |

#### 🧩 Common Confusions
| True Flag | Predicted As | Reason |
|------------|--------------|--------|
| 🇸🇸 South Sudan | 🇩🇯 Djibouti | Similar geometric layout |
| 🇱🇷 Liberia | 🇺🇸 USA | Shared stars & stripes |
| 🇬🇧 United Kingdom | 🇮🇪 Northern Ireland | Union Jack overlap |
| 🇲🇶 Martinique | 🇬🇵 Guadeloupe | Same French tricolor pattern |

---

## 🧩 Future Work
- 🔎 Grad-CAM visualization to highlight model attention regions  
- 🧠 Vision Transformer (ViT) version  
- 📈 Semi-supervised fine-tuning for rare flags  
- 🎨 Synthetic flag generation for augmentation

---

## 🤗 Hugging Face Model Card

### Model Description
Fine-tuned **ResNet-50** model for multi-class flag recognition. Trained on over 200 global flags using PyTorch and torchvision.  
The model outputs a confidence score for each country.

### Intended Use
For educational, research, and demonstration purposes — not for geopolitical classification.

### Example
```python
from PIL import Image
import torch
from torchvision import models, transforms

# Load model
model = models.resnet50()
model.fc = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(model.fc.in_features, 200))
model.load_state_dict(torch.load('flag_classifier_resnet50.pth', map_location='cpu'))
model.eval()

# Predict
img = Image.open('examples/kenya.png')
# (Apply same preprocessing as in training)
```
## 👨‍💻 Author
**Winstan Otieno**  
Machine Learning Engineer & Computer Vision Enthusiast  
📧 win1otieno2@gmail.com/ otienowin@gmail.com
🌐 [GitHub](https://github.com/win1otieno2)
