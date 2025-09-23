

# Importation des modules nécessaires
import cv2
import os
import pandas as pd
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog

# Fenêtre de sélection de la vidéo
video_path = filedialog.askopenfilename(title="Sélectionner la vidéo", filetypes=[("Fichiers vidéo", "*.mp4;*.avi;*.mov;*.mkv")])
if not video_path:
    print("Aucune vidéo sélectionnée.")
    exit()

# Fenêtre de sélection du fichier PKL
pkl_path = filedialog.askopenfilename(
    title="Choisissez un fichier .pkl de comparaison",
    filetypes=[("Fichiers pickle", "*.pkl")]
)
if not pkl_path:
    print("Aucun fichier PKL sélectionné.")
    exit()

# Dossier de sortie pour les frames
output_dir = os.path.join(os.path.dirname(video_path), "frames")
os.makedirs(output_dir, exist_ok=True)

# Extraction des frames
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0
image_files = []
print("Extraction des images...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % 20 == 0:
        filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        image_files.append(f"frame_{frame_count:05d}.jpg")
        saved_count += 1
    frame_count += 1
cap.release()
print(f"{saved_count} images extraites dans {output_dir}")

# Chargement des embeddings
print("\n📂 Chargement de la base d'embeddings...")
embeddings_list = pd.read_pickle(pkl_path)
print(f"✅ Base chargée avec {len(embeddings_list)} visages.")

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Création du fichier de résultats
results_path = os.path.join(output_dir, "Résultats.txt")
with open(results_path, "w", encoding="utf-8") as f_out:
    # Parcours des images extraites
    for img_name in image_files:
        img_path = os.path.join(output_dir, img_name)
        print(f"\n🔍 {img_name} : Calcul de l'embedding...")
        f_out.write(f"\n🔍 {img_name} : Calcul de l'embedding...\n")
        try:
            embedding_test = DeepFace.represent(
                img_path=img_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )
            if isinstance(embedding_test, list):
                embedding_test = embedding_test[0]["embedding"]
            else:
                embedding_test = embedding_test["embedding"]
            embedding_test = np.array(embedding_test)
        except Exception as e:
            print(f"Erreur DeepFace : {e}")
            f_out.write(f"Erreur DeepFace : {e}\n")
            continue

        # Comparaison avec la base
        results = []
        for item in embeddings_list:
            sim = cosine_similarity(embedding_test, item["embedding"])
            results.append((item["identity"], sim))
        results.sort(key=lambda x: x[1], reverse=True)

        # Affichage du meilleur match
        if results:
            identity, sim = results[0]
            name = os.path.basename(os.path.dirname(identity))
            print(f"🏆 Meilleure correspondance : {name} ({sim:.4f})")
            f_out.write(f"🏆 Meilleure correspondance : {name} ({sim:.4f})\n")
        else:
            print("Aucune correspondance trouvée.")
            f_out.write("Aucune correspondance trouvée.\n")
print(f"\nTous les résultats ont été enregistrés dans {results_path}")