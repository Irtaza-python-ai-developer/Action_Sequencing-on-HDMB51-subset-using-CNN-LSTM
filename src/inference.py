import torch
import json
import os
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from models.cnn_lstm import CNN_LSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/model.pth"
LABELS_PATH = "models/labels.json"

# load labels
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# load model
model = CNN_LSTM(num_classes=len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_action(image: Image.Image):
    """
    image: PIL Image
    returns: (annotated_image, predicted_label)
    """

    # create fake sequence (repeat same image 15 times)
    frames = []
    for _ in range(15):
        frames.append(transform(image))

    frames = torch.stack(frames)            # (15, C, H, W)
    frames = frames.unsqueeze(0).to(DEVICE) # (1, 15, C, H, W)

    with torch.no_grad():
        outputs = model(frames)
        pred = torch.argmax(outputs, dim=1).item()

    label = inv_label_map[pred]

    # annotate image
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    draw.text((10, 10), f"Action: {label}", fill="red")

    return annotated, label
