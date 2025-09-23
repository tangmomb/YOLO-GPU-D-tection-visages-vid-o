# YOLO GPU et détection de visages dans une vidéo 😃

## Installation et lancement

1. **Créer un environnement virtuel Python**

   - Ouvrez un terminal dans le dossier du projet.
   - Exécutez :
     ```powershell
     python -m venv venv
     ```

2. **Activer l'environnement virtuel**

   - Sous Windows :
     ```powershell
     .\venv\Scripts\activate
     ```

3. **Installer les dépendances**

   - Exécutez :
     ```powershell
     pip install -r requirements.txt
     ```

4. **Lancer le menu graphique**
   - Double-cliquez sur le fichier `Launcher.bat`.
   - Un menu s'ouvre pour lancer les différents scripts du projet.

---

## Description des scripts

- **Détection YOLO avec GPU**

  > Détection d'objets en direct d'une vidéo avec YOLOv8. Affiche les résultats en temps réel.

- **Extraction de frames d'une vidéo**

  > Extrait une image toutes les 20 frames d'une vidéo choisie. Les images sont enregistrées dans le dossier 'frames'.

- **Comparer visages d'un dossier**

  > Compare chaque image d'un dossier à une base d'embeddings (PKL sous forme de list) avec DeepFace ArcFace/RetinaFace. Affiche le meilleur match.

- **Détection célébrité vidéo**

  > Extrait des images d'une vidéo puis compare chaque image à une base d'embeddings. Résultats enregistrés dans Résultats.txt.

- **Détection visage en live avec OpenCV**

  > Détecte les visages dans une vidéo avec OpenCV (Haar cascade). Affiche les visages détectés en temps réel.

- **Info GPU**

  > Affiche les infos sur le GPU et teste la détection avec PyTorch et YOLO.

- **Utilisation GPU YOLO**
  > Teste si YOLO utilise bien le GPU ou le CPU. Ne veut pas dire qu'il va utiliser le GPU pour toutes les opérations. Juste un script de test.

---

## Remarques

- Assurez-vous d'utiliser Python 3.10 ou 3.11 pour une compatibilité optimale avec les dépendances.
- Les scripts utilisent Tkinter pour l'interface graphique et DeepFace/YOLO pour la détection et la reconnaissance.
- Vous pouvez créer un moèdle YOLO custom en suivant ce super tuto : https://youtu.be/r0RspiLG260?si=IBob4Px6ozpPJsO3
