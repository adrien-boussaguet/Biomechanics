import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- Paramètres à ajuster ---
file_path = 'G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Elbow Tracking.csv'
window_pos = 21  # Lissage de la position (impair, ex: 15, 21, 31)
window_acc = 21  # Lissage de l'accélération (impair)

try:
    df = pd.read_csv(file_path, sep=';', decimal=',', skiprows=0) 
    
    time = df.iloc[:, 0]
    
    ox, oy = df.iloc[:, 7], df.iloc[:, 8]
    ax, ay = df.iloc[:, 9], df.iloc[:, 10]
    bx, by = df.iloc[:, 11], df.iloc[:, 12]

    # 1. Calcul de l'angle
    oa_x, oa_y = ax - ox, ay - oy
    ob_x, ob_y = bx - ox, by - oy

    dot_product = oa_x * ob_x + oa_y * ob_y
    norm_oa = np.sqrt(oa_x**2 + oa_y**2)
    norm_ob = np.sqrt(ob_x**2 + ob_y**2)

    cos_theta = np.clip(dot_product / (norm_oa * norm_ob), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    # 2. Lissage de l'angle
    angle_smoothed = savgol_filter(angle_deg, window_length=window_pos, polyorder=3)
    
    # 3. Vitesse et Accélération angulaires
    dt = np.mean(np.diff(time))
    
    # Vitesse en °/s puis conversion en rad/s pour la physique
    angular_velocity_deg = np.gradient(angle_smoothed, dt)
    angular_velocity_rad = angular_velocity_deg * (np.pi / 180)
    
    # Accélération en rad/s²
    angular_acceleration_raw = np.gradient(angular_velocity_rad, dt)
    
    # Second lissage indispensable pour contrer le bruit de la double dérivation
    angular_acceleration_smoothed = savgol_filter(angular_acceleration_raw, window_length=window_acc, polyorder=2)

    # 4. Affichage graphique à 3 étages
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Graphe 1 : Angle
    ax1.plot(time, angle_deg, alpha=0.3, color='gray', label='Angle brut')
    ax1.plot(time, angle_smoothed, color='red', linewidth=2, label='Angle lissé')
    ax1.set_ylabel('Angle (°)')
    ax1.set_title('Cinématique articulaire')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Graphe 2 : Vitesse
    ax2.plot(time, angular_velocity_rad, color='blue', linewidth=2, label='Vitesse (rad/s)')
    ax2.set_ylabel('Vitesse (rad/s)')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Graphe 3 : Accélération
    ax3.plot(time, angular_acceleration_raw, alpha=0.3, color='gray', label='Accélération brute')
    ax3.plot(time, angular_acceleration_smoothed, color='green', linewidth=2, label='Accélération lissée')
    ax3.set_ylabel('Accélération (rad/s²)')
    ax3.set_xlabel('Temps (s)')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Erreur : {e}")