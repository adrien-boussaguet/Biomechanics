import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 1. Paramètres Biomécaniques ---
F0 = 1000.0      # Force isométrique max (N)
a = 250.0        # Constante a (N)
b = 0.25         # Constante b (m/s)
L_opt = 0.3      # Longueur optimale (m)
W = 0.1          # Largeur de la cloche force-longueur
k_passif = 2000.0 # Raideur passive

# Variables de la simulation
Charge = 300.0   # Charge à soulever (N)
Activation = 1.0 # Activation maximale
L_initial = 0.4  # Le muscle est initialement étiré à 40 cm

# --- 2. Fonctions Physiologiques ---

def force_longueur_active(L):
    """ Courbe en cloche de la force active """
    return np.exp(-((L - L_opt) / W)**2)

def force_passive(L):
    """ Résistance exponentielle à l'étirement """
    if L <= L_opt:
        return 0.0
    else:
        return k_passif * (L - L_opt)**2

def dynamique_muscle(t, L):
    """ 
    Équation différentielle dL/dt = -V
    (Le signe moins car V>0 signifie un raccourcissement) 
    """
    # 1. Calcul de la force que l'élément contractile DOIT fournir
    F_ce = Charge - force_passive(L)
    
    # 2. Force isométrique maximale à cette longueur précise
    F_iso_actuelle = F0 * Activation * force_longueur_active(L)
    
    # Si la charge est supérieure à ce que le muscle peut faire (en concentrique)
    if F_ce >= F_iso_actuelle:
        return 0.0 # Simplification : on bloque le mouvement (isométrique)

    # 3. Équation de Hill inversée pour trouver la vitesse (V)
    # V = b * (F0_actuelle - F_ce) / (F_ce + a * Activation * f_L)
    numerateur = F_iso_actuelle - F_ce
    denominateur = F_ce + (a * Activation * force_longueur_active(L))
    
    V = b * (numerateur / denominateur)
    
    return -V # dL/dt est négatif si le muscle se raccourcit

# --- 3. Simulation (Résolution de l'EDO) ---

# Intervalle de temps : de 0 à 1.5 secondes
t_span = (0, 1.5)

# solve_ivp résout le système. 
# y0=[L_initial] est la condition initiale.
resultat = solve_ivp(dynamique_muscle, t_span, y0=[L_initial], 
                     t_eval=np.linspace(0, 1.5, 200), method='RK45')

temps = resultat.t
longueurs = resultat.y[0]

# --- 4. Affichage des résultats ---

plt.figure(figsize=(10, 5))

# Graphique de la cinématique
plt.plot(temps, longueurs, 'r-', linewidth=2)
plt.axhline(y=L_opt, color='k', linestyle='--', label='Longueur optimale ($L_{opt}$)')

plt.title('Contraction Isotonique : Longueur du muscle vs Temps')
plt.xlabel('Temps (s)')
plt.ylabel('Longueur du muscle (m)')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()