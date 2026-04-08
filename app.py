from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

# ── Load Model Once at Startup ────────────────────────────
checkpoint = torch.load('model.pth', map_location='cpu')
CLASS_NAMES = checkpoint['class_names']

model = models.efficientnet_b0(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(1280, len(CLASS_NAMES))
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# ── Preprocessing ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Routes ────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    tensor = transform(img).unsqueeze(0)  # Add batch dim

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top3 = torch.topk(probs, 3)  # Top 3 predictions

    results = []
    for i in range(3):
        results.append({
            'breed': CLASS_NAMES[top3.indices[i]].replace('_', ' ').title(),
            'confidence': round(top3.values[i].item() * 100, 2)
        })

    return jsonify({'predictions': results})

if __name__ == '__main__':
    app.run(debug=True)