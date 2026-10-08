# detection/fire/detector.py
import torch
from torchvision import transforms, models
from PIL import Image

CFG = {
    "img_size": 224,
    "checkpoint": "fire/checkpoint/fire_model.pt",
    "num_classes": 2,
    "label_map": {0: "fire", 1: "normal"}
}

def load_model(path=CFG["checkpoint"], num_classes=CFG["num_classes"]):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor()
])

def detect(image_path, model=None):
    model = model or load_model()
    img = Image.open(image_path).convert("RGB")
    t = _transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
        pred = int(out.argmax(dim=1).item())
        score = float(torch.softmax(out, dim=1)[0, pred])
    return {"label": CFG["label_map"][pred], "score": score}