"""
Script de détection en direct avec YOLOv8
- Fonctionne sur webcam OU vidéo
- Si c'est une vidéo : lecture en vitesse réelle (selon FPS du fichier)
- Si c'est une webcam : lecture en temps réel (pas de sleep)
"""

import time
import cv2
from ultralytics import YOLO
import torch
import tkinter as tk
from tkinter import filedialog

# -------------------------------
# CONFIGURATION UTILISATEUR
# -------------------------------

## Fenêtre pour choisir le fichier vidéo (0 = webcam)
SOURCE = filedialog.askopenfilename(title="Sélectionne la vidéo à analyser", filetypes=[("Fichiers vidéo", "*.mp4;*.avi;*.mov;*.mkv")])
if not SOURCE:
    print("Aucune vidéo sélectionnée. Arrêt du script.")
    exit()
MODEL_NAME = "yolov8x.pt"   # Choisir taille du modèle : n, s, m, l, x
CONF_THRESH = 0.35          # Seuil de confiance minimal pour afficher une détection

# -------------------------------
# INITIALISATION
# -------------------------------

# Vérifie si GPU dispo
device = "cuda" if torch.cuda.is_available() else "cpu"
print("📦 Utilisation du device :", device)

# Charge le modèle
model = YOLO(MODEL_NAME)
# Forcer l'utilisation du GPU si disponible
if torch.cuda.is_available():
    model.to("cuda")
    print("✅ YOLO tourne maintenant sur le GPU :", torch.cuda.get_device_name(0))
else:
    print("⚠️ Pas de GPU détecté, utilisation du CPU")

# Ouvre la source vidéo
cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"Impossible d’ouvrir la source : {SOURCE}")

# Récupère FPS si c’est un fichier vidéo
# (avec une webcam ça renvoie souvent 0 ou un nombre incorrect)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = 0
if fps > 0:
    delay = 1 / fps  # durée d'une frame en secondes
    print(f"🎬 Vidéo détectée ({fps:.1f} FPS), lecture en vitesse réelle")
else:
    print("📹 Webcam détectée, lecture en temps réel")

# -------------------------------
# BOUCLE PRINCIPALE
# -------------------------------

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Fin du flux ou erreur de lecture")
        break

    # Détection
    results = model(frame, conf=CONF_THRESH, verbose=False)[0]
    annotated_frame = results.plot()

    # --- Redimensionnement proportionnel (largeur fixe = 1000 px) ---
    TARGET_WIDTH = 1000
    h, w = annotated_frame.shape[:2]
    scale = TARGET_WIDTH / w
    new_h = int(h * scale)
    resized_frame = cv2.resize(annotated_frame, (TARGET_WIDTH, new_h))

    # Affichage
    cv2.imshow("YOLO Live / Q pour quitter", resized_frame)

    # Quitter
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # --- Correction vitesse de lecture ---
    if delay > 0:  # uniquement si c'est une vidéo
        elapsed = time.time() - prev_time
        to_wait = delay - elapsed
        if to_wait > 0:
            time.sleep(to_wait)
        prev_time = time.time()


# -------------------------------
# FERMETURE
# -------------------------------

cap.release()
cv2.destroyAllWindows()
print("✅ Fermeture propre du script")
