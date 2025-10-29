import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageOps

# --- CONFIG ---
DATA_DIR = Path(r"C:\Learn\ML\Classification\flags\flags_dataset")
MODEL_PATH = Path("flag_classifier_resnet50.pth")
BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- WHITE BACKGROUND FIX ---
def white_background(img: Image.Image) -> Image.Image:
    """Ensures all flags have white background instead of transparency/black."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])  # alpha composite
        return bg
    return img.convert("RGB")

# Custom loader applying background fix
def pil_loader_with_white_bg(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return white_background(img)

# --- TRANSFORMS ---
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=3),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomPerspective(distortion_scale=0.05, p=0.3),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- DATASETS ---
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms, loader=pil_loader_with_white_bg)
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=test_transforms, loader=pil_loader_with_white_bg)

# --- BALANCED SAMPLING ---
class_counts = Counter([label for _, label in train_dataset.samples])
weights = 1.0 / np.array([class_counts[label] for _, label in train_dataset.samples])
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- MODEL ---
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
for param in model.parameters():
    param.requires_grad = True  # Fine-tune all layers

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, len(train_dataset.classes))
)
model = model.to(DEVICE)

# --- TRAINING SETUP ---
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- TRAIN LOOP ---
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()
    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/total:.4f} Acc: {acc:.2f}%")

# --- SAVE MODEL + CLASSES ---
torch.save(model.state_dict(), MODEL_PATH)
with open("class_names.txt", "w", encoding="utf-8") as f:
    for c in train_dataset.classes:
        f.write(c + "\n")

print(f"✅ Training complete. Model saved to {MODEL_PATH}")
