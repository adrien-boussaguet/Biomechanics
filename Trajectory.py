import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Chemin vers ton fichier exporté
file_path = 'G:\Documents\VS Code\Biomecha\Biomechanics\VID_20260813_150533.csv'

try:
    # Lecture du CSV 
    # (Si erreur de colonnes, essaie sep='\t' au lieu de sep=';')
    df = pd.read_csv(file_path, sep=';', decimal=',', skiprows=0) 
    
    # Extraction des colonnes via leur position (iloc) d'après ta capture
    time = df.iloc[:, 0]  # Colonne 0 : Temps
    
    # Coordonnées du sommet O (origine de l'angle)
    ox = df.iloc[:, 7]    # Angle 1/o/X
    oy = df.iloc[:, 8]    # Angle 1/o/Y
    
    # Coordonnées du point A (première extrémité)
    ax = df.iloc[:, 9]    # Angle 1/a/X
    ay = df.iloc[:, 10]   # Angle 1/a/Y
    
    # Coordonnées du point B (seconde extrémité)
    bx = df.iloc[:, 11]   # Angle 1/b/X
    by = df.iloc[:, 12]   # Angle 1/b/Y

    # 1. Mathématiques : Calcul des vecteurs OA et OB
    oa_x, oa_y = ax - ox, ay - oy
    ob_x, ob_y = bx - ox, by - oy

    # 2. Calcul du produit scalaire et des normes
    dot_product = oa_x * ob_x + oa_y * ob_y
    norm_oa = np.sqrt(oa_x**2 + oa_y**2)
    norm_ob = np.sqrt(ob_x**2 + ob_y**2)

    # 3. Déduction de l'angle en radians puis conversion en degrés
    # Utilisation de np.clip pour éviter les erreurs mathématiques liées aux arrondis
    cos_theta = np.clip(dot_product / (norm_oa * norm_ob), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    # 4. Lissage avec Savitzky-Golay
    # window_length (impair) gère la force du lissage. 15 ou 21 sont de bonnes bases.
    angle_smoothed = savgol_filter(angle_deg, window_length=15, polyorder=3)
    
    # 5. Calcul de la vitesse angulaire (optionnel mais utile)
    dt = np.mean(np.diff(time))
    angular_velocity = np.gradient(angle_smoothed, dt)

    # 6. Affichage graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Graphe 1 : Évolution de l'angle
    ax1.plot(time, angle_deg, label='Angle brut calculé', alpha=0.4, color='gray')
    ax1.plot(time, angle_smoothed, label='Angle lissé (Savitzky-Golay)', color='red', linewidth=2)
    ax1.set_ylabel('Angle (Degrés)')
    ax1.set_title('Cinématique articulaire : Évolution et lissage de l\'angle')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Graphe 2 : Vitesse angulaire
    ax2.plot(time, angular_velocity, color='blue', label='Vitesse angulaire (°/s)')
    ax2.set_ylabel('Vitesse (°/s)')
    ax2.set_xlabel('Temps')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Erreur lors du traitement : {e}")
    print("Vérifie le nom du fichier, le séparateur (sep) et assure-toi que les colonnes 7 à 12 existent bien.")