nvidia-smi
import torch
print("CUDA Available:", torch.cuda.is_available())

!pip install ultralytics

import torch
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load EfficientNet-B0 pretrained on ImageNet
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
efficientnet = efficientnet_b0(weights=weights).to(device)
efficientnet.eval()

# Load YOLO
yolo = YOLO("yolov5s.pt")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def prepare_image(path):
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)

def run_models(image_path):
    x = prepare_image(image_path)

    # EfficientNet
    with torch.no_grad():
        logits = efficientnet(x)
        probs = torch.softmax(logits[0], dim=0)
        effnet_conf = torch.max(probs).item()

    # YOLO
    yolo_out = yolo(image_path)
    yolo_conf = max([box.conf for box in yolo_out[0].boxes], default=0)

    return effnet_conf, yolo_conf

def consensus(effnet_conf, yolo_conf, threshold=0.3):
    if abs(effnet_conf - yolo_conf) > threshold:
        return "Potential Adversarial Attack"
    else:
        return "Normal Image"

image = "/content/pexels-cottonbro-studio-5473956-min_11zon.jpg"
e_conf, y_conf = run_models(image)
print("EfficientNet Confidence:", e_conf)
print("YOLO Confidence:", y_conf)
print("Decision:", consensus(e_conf, y_conf))

image = "/content/british-wild-cat-1.jpg"
e_conf, y_conf = run_models(image)
print("EfficientNet Confidence:", e_conf)
print("YOLO Confidence:", y_conf)
print("Decision:", consensus(e_conf, y_conf))
