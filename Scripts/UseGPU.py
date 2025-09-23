from ultralytics import YOLO
import torch

# Charger modèle
model = YOLO("yolov8n.pt")

# Vérifier si GPU dispo et forcer l'utilisation
if torch.cuda.is_available():
    model.to("cuda")
    print("✅ YOLO tourne maintenant sur le GPU :", torch.cuda.get_device_name(0))
else:
    print("⚠️ Pas de GPU détecté, utilisation du CPU")
