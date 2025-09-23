import torch

print("----- Torch Info -----")
print("PyTorch version :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Nombre de GPU :", torch.cuda.device_count())
    print("Nom du GPU :", torch.cuda.get_device_name(0))
    print("Version CUDA :", torch.version.cuda)
    print("Device courant :", torch.cuda.current_device())
else:
    print("⚠️  Aucun GPU détecté par PyTorch !")

print("\n----- Ultralytics YOLO Test -----")
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    device = model.device
    print("YOLO utilise :", device)
except Exception as e:
    print("Erreur YOLO :", e)
