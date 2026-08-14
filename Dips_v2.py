import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter

#Masses
m_body = 80     #kg - masse du sujet
m_weight = 80   #kg - charge ajoutée (disques sur ceinture lestée)

m_forearm = 2 * 0.016 * m_body   #avant-bras (2x), coeff Winter
m_arm     = 2 * 0.028 * m_body   #bras (2x), coeff Winter
m_bust    = m_body - m_forearm - m_arm   #tronc+tête+jambes SEULEMENT (poids ajouté traité à part, cf plus bas)

g = 9.81

#Segments
L1 = 0.272  #m - avant-bras
L2 = 0.31   #m - bras
d3 = 0.60   #m - épaule -> CdM du buste (tronc+tête+jambes)

#--- Données anthropométriques (Winter, "Biomechanics and Motor Control of
#    Human Movement") : position du CdM en fraction de la longueur du
#    segment depuis l'extrémité PROXIMALE ANATOMIQUE (coude pour l'avant-bras,
#    épaule pour le bras), et rayon de giration autour du CdM en fraction
#    de la longueur du segment.
#
#    ATTENTION : dans ce script, la chaîne cinématique part de la MAIN
#    (point fixe), donc son extrémité "proximale" (côté base de la chaîne)
#    est l'OPPOSÉ de l'extrémité proximale anatomique pour l'avant-bras et
#    le bras (main <-> épaule sont inversés par rapport à la convention
#    anatomique habituelle). Il faut donc utiliser (1 - fraction_anthropo)
#    pour repérer le CdM depuis le point fixe de la chaîne.
ANTHRO_FOREARM_COM = 0.430   #depuis le coude (proximal anatomique)
ANTHRO_FOREARM_GYR = 0.303   #rayon de giration / longueur, autour du CdM
ANTHRO_ARM_COM = 0.436       #depuis l'épaule (proximal anatomique)
ANTHRO_ARM_GYR = 0.322

r1 = 1 - ANTHRO_FOREARM_COM   #CdM avant-bras, fraction de L1 depuis la MAIN
r2 = 1 - ANTHRO_ARM_COM       #CdM bras, fraction de L2 depuis le COUDE

#Inertie (I_cm, autour du centre de masse de chaque segment)
I_cm_forearm = m_forearm * (ANTHRO_FOREARM_GYR * L1) ** 2
I_cm_arm     = m_arm * (ANTHRO_ARM_GYR * L2) ** 2
I_cm_bust    = 42.5   #kg.m^2 - valeur de départ (ex "I_shoulder"), à affiner
                       #quand le buste sera décomposé tronc/jambes (point 4, en attente)

#Bras de levier musculaire (réservés pour une future estimation de force
#musculaire F = tau / la -- non utilisés dans le calcul de couple ci-dessous)
la_elbow = 0.025    #m
la_shoulder = 0.045 #m

#--- Position de la charge ajoutée --------------------------------------
#La ceinture lestée pend le long du corps jusqu'à hauteur quasi-des-pieds
#(sujet de 1.86 m) : on modélise donc le poids comme une masse ponctuelle
#à distance d_weight de l'épaule, alignée sur le même angle theta3 que le
#buste (hypothèse : chaîne tendue, suit globalement l'axe tronc-jambes).
h_athlete = 1.86     #m - taille du sujet
frac_shoulder = 0.818  #hauteur épaule / taille (Winter/de Leva)
frac_ankle = 0.039     #hauteur cheville / taille
d_weight = (frac_shoulder - frac_ankle) * h_athlete   #~1.45 m épaule -> chevilles
#(ajuste d_weight si les disques pendent plus haut/bas que la cheville)

#Extraction CSV
files_path = ['G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Hand Tracking.csv', 'G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Elbow Tracking.csv', 'G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Shoulder Trackingv4 TOP2.csv']

list_angle_rad = []
dt = None

#extraction des csv
for i in range(len(files_path)):
    df = pd.read_csv(files_path[i], sep=';', decimal=',', skiprows=0)

    time = df.iloc[:, 0]
    if dt is None:
        dt = float(np.mean(np.diff(time)))  #pas de temps réel (issu du CSV), pour les dérivées

    ox, oy = df.iloc[:, 7], df.iloc[:, 8]
    ax, ay = df.iloc[:, 9], df.iloc[:, 10]
    bx, by = df.iloc[:, 11], df.iloc[:, 12]

    oa_x, oa_y = ax - ox, ay - oy
    ob_x, ob_y = bx - ox, by - oy

    dot_product = oa_x * ob_x + oa_y * ob_y
    det = oa_x * ob_y - oa_y * ob_x

    angle_rad = np.arctan2(det, dot_product)

    list_angle_rad.append(angle_rad)

All_angle_rad = pd.concat(list_angle_rad, axis=1)

#déroulement de l'angle (np.unwrap) AVANT tout lissage/dérivation, pour
#éviter un artefact si l'angle traverse +-pi pendant le mouvement
All_angle_unwrapped = All_angle_rad.apply(lambda col: np.unwrap(col.to_numpy()), axis=0)

def savgol_col(col, deriv):
    return savgol_filter(col, window_length=11, polyorder=3, deriv=deriv, delta=dt)

All_angle_smooth = All_angle_unwrapped.apply(lambda c: savgol_col(c, 0), axis=0)  #angle
All_angle_vel    = All_angle_unwrapped.apply(lambda c: savgol_col(c, 1), axis=0)  #vitesse angulaire
All_angle_acc    = All_angle_unwrapped.apply(lambda c: savgol_col(c, 2), axis=0)  #accélération angulaire
print(All_angle_smooth)

#calcul des angles absolus (relatif à la vertical) -- theta
theta1 = - All_angle_smooth.iloc[:, 0] + np.pi / 2
theta2 = + theta1 - np.pi + All_angle_smooth.iloc[:, 1]
theta3 = np.pi - (- theta2 + All_angle_smooth.iloc[:, 2])

#mêmes combinaisons linéaires, appliquées aux dérivées (les constantes
#pi/2, pi disparaissent en dérivant -- valide car la dérivation est linéaire)
omega1 = - All_angle_vel.iloc[:, 0]
alpha1 = - All_angle_acc.iloc[:, 0]
omega2 = omega1 + All_angle_vel.iloc[:, 1]
alpha2 = alpha1 + All_angle_acc.iloc[:, 1]
omega3 = omega2 - All_angle_vel.iloc[:, 2]
alpha3 = alpha2 - All_angle_acc.iloc[:, 2]

n = len(theta1)

#--- Cinématique : position, vitesse, accélération de chaque point ------
def dir_e(theta):
    return np.column_stack((np.sin(theta), np.cos(theta)))

def dir_e_perp(theta):
    return np.column_stack((np.cos(theta), -np.sin(theta)))

def point_kin(P_base, V_base, A_base, r, theta, omega, alpha):
    """Cinématique d'un point situé à distance r (constante) le long de
    l'angle absolu theta, depuis une base elle-même en mouvement."""
    theta = np.asarray(theta); omega = np.asarray(omega); alpha = np.asarray(alpha)
    e = dir_e(theta)
    e_perp = dir_e_perp(theta)
    P = P_base + r * e
    V = V_base + r * omega[:, None] * e_perp
    A = A_base + r * alpha[:, None] * e_perp - r * (omega ** 2)[:, None] * e
    return P, V, A

P_hand = np.zeros((n, 2)); V_hand = np.zeros((n, 2)); A_hand = np.zeros((n, 2))  #main = base fixe

P_elbow, V_elbow, A_elbow = point_kin(P_hand, V_hand, A_hand, L1, theta1, omega1, alpha1)
P_cm1, _, A_cm1 = point_kin(P_hand, V_hand, A_hand, r1 * L1, theta1, omega1, alpha1)      #CdM avant-bras

P_shoulder, V_shoulder, A_shoulder = point_kin(P_elbow, V_elbow, A_elbow, L2, theta2, omega2, alpha2)
P_cm2, _, A_cm2 = point_kin(P_elbow, V_elbow, A_elbow, r2 * L2, theta2, omega2, alpha2)   #CdM bras

G_bust, _, A_cm3 = point_kin(P_shoulder, V_shoulder, A_shoulder, d3, theta3, omega3, alpha3)          #CdM buste
P_weight, _, A_weight = point_kin(P_shoulder, V_shoulder, A_shoulder, d_weight, theta3, omega3, alpha3)  #charge

#--- Dynamique inverse (Newton-Euler récursif, 2D) -----------------------
def cross2(r_vec, F_vec):
    return r_vec[:, 0] * F_vec[:, 1] - r_vec[:, 1] * F_vec[:, 0]

def moment_about(P_joint, parts):
    """Couple total, autour de P_joint, dû à tous les segments/masses
    situés au-delà de cette articulation.
    parts: liste de tuples (masse, I_cm, P_cm, A_cm, alpha)."""
    g_vec = np.array([0.0, -g])
    total = np.zeros(P_joint.shape[0])
    for m, I_cm, P_cm, A_cm, alpha in parts:
        r_vec = P_cm - P_joint
        F_eff = m * (A_cm - g_vec)          #force "effective" (inertie - gravité)
        total += I_cm * np.asarray(alpha) + cross2(r_vec, F_eff)
    return total

tau_shoulder = moment_about(P_shoulder, [
    (m_bust,   I_cm_bust, G_bust,    A_cm3,   alpha3),
    (m_weight, 0.0,       P_weight,  A_weight, alpha3),
])

tau_elbow = moment_about(P_elbow, [
    (m_arm,    I_cm_arm,  P_cm2,     A_cm2,   alpha2),
    (m_bust,   I_cm_bust, G_bust,    A_cm3,   alpha3),
    (m_weight, 0.0,       P_weight,  A_weight, alpha3),
])

tau_hand = moment_about(P_hand, [
    (m_forearm, I_cm_forearm, P_cm1,    A_cm1,   alpha1),
    (m_arm,     I_cm_arm,     P_cm2,    A_cm2,   alpha2),
    (m_bust,    I_cm_bust,    G_bust,   A_cm3,   alpha3),
    (m_weight,  0.0,          P_weight, A_weight, alpha3),
])

n_frames = len(theta1)
frames_x = np.arange(n_frames)

#Config plots
fig, (ax_anim, ax_torque) = plt.subplots(1, 2, figsize=(12, 5))

# Graphique 1 : Squelette et trajectoires
ax_anim.set_xlim(-0.5, 1)
ax_anim.set_ylim(-1, 1)
ax_anim.set_aspect("equal")
ax_anim.set_title("Cinématique du mouvement")
ax_anim.set_xlabel("X (m)")
ax_anim.set_ylabel("Y (m)")
ax_anim.grid(True, alpha=0.3)

(ligne_corps,) = ax_anim.plot([], [], "ko-", lw=3, markersize=8, label="Segments")
(trajectoire_coude,) = ax_anim.plot([], [], "r--", lw=1, alpha=0.5, label="Coude")
(trajectoire_epaule,) = ax_anim.plot([], [], "g--", lw=1, alpha=0.5, label="Épaule")
(trajectoire_buste,) = ax_anim.plot([], [], "b--", lw=1, alpha=0.5, label="Buste")
ax_anim.legend(loc="upper right")

# Graphique 2 : Torques superposés
ax_torque.set_xlim(0, n_frames)
y_min = min(tau_hand.min(), tau_elbow.min(), tau_shoulder.min())
y_max = max(tau_hand.max(), tau_elbow.max(), tau_shoulder.max())
margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
ax_torque.set_ylim(y_min - margin, y_max + margin)

ax_torque.set_title("Couples articulaires instantanés")
ax_torque.set_xlabel("Frames / Temps")
ax_torque.set_ylabel(r"Torque ($\mathrm{N}\cdot\mathrm{m}$)")
ax_torque.grid(True, alpha=0.3)

(line_tau_hand,) = ax_torque.plot([], [], "m-", lw=2, label="Torque Main")
(line_tau_elbow,) = ax_torque.plot([], [], "r-", lw=2, label="Torque Coude")
(line_tau_shoulder,) = ax_torque.plot([], [], "g-", lw=2, label="Torque Épaule")
ax_torque.legend(loc="upper right")

plt.tight_layout()


#Animation synchro
def animate(i):
    x = [P_hand[i, 0], P_elbow[i, 0], P_shoulder[i, 0], G_bust[i, 0]]
    y = [P_hand[i, 1], P_elbow[i, 1], P_shoulder[i, 1], G_bust[i, 1]]
    ligne_corps.set_data(x, y)

    trajectoire_coude.set_data(P_elbow[: i + 1, 0], P_elbow[: i + 1, 1])
    trajectoire_epaule.set_data(P_shoulder[: i + 1, 0], P_shoulder[: i + 1, 1])
    trajectoire_buste.set_data(G_bust[: i + 1, 0], G_bust[: i + 1, 1])

    t = frames_x[: i + 1]
    line_tau_hand.set_data(t, tau_hand[: i + 1])
    line_tau_elbow.set_data(t, tau_elbow[: i + 1])
    line_tau_shoulder.set_data(t, tau_shoulder[: i + 1])

    return (
        ligne_corps,
        trajectoire_coude,
        trajectoire_epaule,
        trajectoire_buste,
        line_tau_hand,
        line_tau_elbow,
        line_tau_shoulder,
    )


ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=30, blit=True)

plt.show()
