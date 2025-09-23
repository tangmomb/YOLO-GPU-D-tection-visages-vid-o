
import cv2
import tkinter as tk
from tkinter import filedialog

# 1. Charger le classifieur pré-entraîné (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# 2. Fenêtre de sélection de la vidéo
video_path = filedialog.askopenfilename(title="Sélectionner la vidéo", filetypes=[("Fichiers vidéo", "*.mp4;*.avi;*.mov;*.mkv")])
if not video_path:
    print("Aucune vidéo sélectionnée.")
    exit()
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la vidéo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # fin de la vidéo

    # 3. Convertir en niveaux de gris (plus rapide pour la détection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. Détecter les visages
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # 5. Dessiner un rectangle autour de chaque visage
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 6. Afficher le résultat
    cv2.imshow("Détection de visages", frame)

    # 7. Quitter avec 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
