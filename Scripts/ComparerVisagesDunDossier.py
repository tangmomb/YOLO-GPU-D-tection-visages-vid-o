
# Importation des modules nécessaires
import cv2
import os
import pandas as pd
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog


# Fenêtre de sélection du dossier d'images extraites
frames_dir = filedialog.askdirectory(title="Sélectionner le dossier des images extraites (frames)")
if not frames_dir:
    print("Aucun dossier sélectionné.")
    exit()

# Fenêtre de sélection du fichier PKL
pkl_path = filedialog.askopenfilename(
    title="Choisissez un fichier .pkl de comparaison",
    filetypes=[("Fichiers pickle", "*.pkl")]
)
if not pkl_path:                                     
    print("Aucun fichier PKL sélectionné.")
    exit()

# Chargement des embeddings
print("📂 Chargement de la base d'embeddings...")
embeddings_list = pd.read_pickle(pkl_path)
print(f"✅ Base chargée avec {len(embeddings_list)} visages.")

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Parcours des images du dossier frames
image_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for img_name in image_files:
    img_path = os.path.join(frames_dir, img_name)
    print(f"\n🔍 {img_name} : Calcul de l'embedding...")
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
    else:
        print("Aucune correspondance trouvée.")