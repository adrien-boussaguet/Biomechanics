import numpy as np
import matplotlib.pyplot as plt

# 1. Création de la ligne du temps (3 secondes, 100 points de calcul)
temps = np.linspace(0, 3.0, 100)

# 2. Définition des angles limites en DEGRÉS (pour la lisibilité)
# Avant-bras (fixe sur les barres)
theta1_deg_haut = -5.0
theta1_deg_bas = -5.0

# Bras (extension -> flexion)
theta2_deg_haut = 10.0
theta2_deg_bas = 100.0

# Buste (droit -> penché en avant)
theta3_deg_haut = 180.0
theta3_deg_bas = 160.0

# 3. Fonction pour créer la trajectoire fluide en "U" (descente puis montée)
def generer_trajectoire(angle_haut, angle_bas, t):
    # La formule cosinus permet un départ arrêté, une accélération, 
    # un ralentissement en bas, et une remontée fluide.
    return angle_bas + (angle_haut - angle_bas) * (1 + np.cos(2 * np.pi * t / 3.0)) / 2

# Génération des courbes en degrés
theta1_deg_traj = generer_trajectoire(theta1_deg_haut, theta1_deg_bas, temps)
theta2_deg_traj = generer_trajectoire(theta2_deg_haut, theta2_deg_bas, temps)
theta3_deg_traj = generer_trajectoire(theta3_deg_haut, theta3_deg_bas, temps)

# (Optionnel mais crucial pour la suite : conversion en radians pour la physique)
theta1_rad_traj = np.radians(theta1_deg_traj)
theta2_rad_traj = np.radians(theta2_deg_traj)
theta3_rad_traj = np.radians(theta3_deg_traj)

# 4. Tracé du graphique
plt.figure(figsize=(10, 5))

plt.plot(temps, theta1_deg_traj, label="Avant-bras (theta 1)", color='blue', linewidth=2)
plt.plot(temps, theta2_deg_traj, label="Bras (theta 2)", color='red', linewidth=2)
plt.plot(temps, theta3_deg_traj, label="Buste (theta 3)", color='green', linewidth=2)

plt.title("Trajectoires angulaires lors d'un Dips (3 secondes)")
plt.xlabel("Temps (secondes)")
plt.ylabel("Angle (degrés)")
plt.axvline(x=1.5, color='grey', linestyle='--', label="Point le plus bas")
plt.legend()
plt.grid(True)
plt.show()