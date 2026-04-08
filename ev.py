import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, 
                              confusion_matrix, 
                              ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

# ── Load test data ────────────────────────────────────────
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_data   = datasets.ImageFolder('dataset_split/test', transform=test_tf)
test_loader = DataLoader(test_data, batch_size=16)
CLASS_NAMES = test_data.classes

# ── Load model ────────────────────────────────────────────
checkpoint = torch.load('model.pth', map_location='cpu')
model = models.efficientnet_b0(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(1280, len(CLASS_NAMES))
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# ── Get predictions ───────────────────────────────────────
all_preds  = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# ── Print F1, Precision, Recall ───────────────────────────
print(classification_report(
    all_labels, all_preds,
    target_names=CLASS_NAMES
))

# ── Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Confusion Matrix - Dog Breed Classifier")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved!")