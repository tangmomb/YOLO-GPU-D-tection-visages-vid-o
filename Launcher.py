import sys
import tkinter as tk
import subprocess
import os



# Liste des scripts et explications

Dossier = os.path.dirname(os.path.abspath(__file__)) + "/Scripts/"

SCRIPTS = [
    {
        "name": "Détection YOLO avec GPU",
        "file": Dossier + "DetectionGPU.py",
        "desc": "Détection d'objets en direct (webcam ou vidéo) avec YOLOv8. Affiche les résultats en temps réel."
    },
    {
        "name": "Extraction de frames d'une vidéo",
        "file": Dossier + "ExtraireCapturesDunFilm.py",
        "desc": "Extrait une image toutes les 20 frames d'une vidéo choisie. Les images sont enregistrées dans le dossier 'frames'."
    },
    {
        "name": "Comparer visages d'un dossier",
        "file": Dossier + "ComparerVisagesDunDossier.py",
        "desc": "Compare chaque image d'un dossier à une base d'embeddings (PKL sous forme de list) avec DeepFace ArcFace/RetinaFace. Affiche le meilleur match."
    },
    {
        "name": "Détection célébrité vidéo",
        "file": Dossier + "DetectionCelebriteVideo.py",
        "desc": "Extrait des images d'une vidéo puis compare chaque image à une base d'embeddings. Résultats enregistrés dans Résultats.txt."
    },
    {
        "name": "Détection visage en live avec OpenCV",
        "file": Dossier + "VisageOpenCV.py",
        "desc": "Détecte les visages dans une vidéo avec OpenCV (Haar cascade). Affiche les visages détectés en temps réel."
    },
    {
        "name": "Info GPU",
        "file": Dossier + "InfoGPU.py",
        "desc": "Affiche les infos sur le GPU et teste la détection avec PyTorch et YOLO."
    },
    {
        "name": "Utilisation GPU YOLO",
        "file": Dossier + "UseGPU.py",
        "desc": "Teste si YOLO utilise bien le GPU ou le CPU. Ne veut pas dire qu'il va utiliser le GPU pour toutes les opérations. Juste un script de test."
    }
]

# Fonction pour lancer un script
def run_script(script_file):
    script_path = os.path.join(os.path.dirname(__file__), script_file)
    subprocess.Popen([sys.executable, script_path])

# Interface Tkinter
root = tk.Tk()
root.title("Launcher")

# Choix de la taille de la fenêtre
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
root.configure(bg="black")

# Choix du padding entre les blocs
FRAME_PADY = 8

for script in SCRIPTS:
    frame = tk.Frame(root, pady=FRAME_PADY, bg="black")
    frame.pack(fill="x")
    btn_text = f"{script['name']}\n\n{script['desc']}"
    btn = tk.Button(frame, text=btn_text, command=lambda f=script["file"]: run_script(f), height=4, width=60, bg="black", fg="white", activebackground="gray20", activeforeground="white", justify="left", anchor="w", wraplength=WINDOW_WIDTH-50)
    btn.pack(anchor="w", padx=10, fill="x")

root.mainloop()
