import cv2
import os
import tkinter as tk
from tkinter import filedialog

# Fenêtre de sélection de la vidéo
video_path = filedialog.askopenfilename(title="Sélectionner la vidéo", filetypes=[("Fichiers vidéo", "*.mp4;*.avi;*.mov;*.mkv")])
if not video_path:
    print("Aucune vidéo sélectionnée.")
    exit()

# Dossier de sortie
output_dir = os.path.join(os.path.dirname(video_path), "frames")
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % 20 == 0:
        filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"{saved_count} images extraites dans {output_dir}")