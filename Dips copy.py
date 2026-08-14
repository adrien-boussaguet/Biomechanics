import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter

#Masses
m_body = 80 #kg
m_weight = 60 #kg

m_forearm = 2 * 0.016 * m_body
m_arm = 2 * 0.028 * m_body

m_legs= 2 * 0.161 * m_body
m_bust = m_body - m_forearm - m_arm - m_legs
m_lower = m_legs + m_weight

g=9.81

#Segments
L1=0.27 #m - avant-bras
L2=0.31 #m - bras
L3=0.45   #m - buste
d3=0.3   #m - centre de masse corps

anthro_forearm_com = 0.430
anthro_forearm_gyr = 0.303
anthro_arm_com = 0.436
anthro_arm_gyr = 0.322

r1 = 1 - anthro_forearm_com
r2 = 1 - anthro_arm_com

#inertie et bras de levier
I_cm_forearm = m_forearm * (anthro_forearm_gyr * L1) ** 2
I_cm_arm = m_arm * (anthro_arm_gyr * L2) ** 2
I_cm_bust = 2.5

la_elbow = 0.025
la_shoulder = 0.045

#Extraction CSV
files_path = ['G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Hand Tracking.csv', 'G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Elbow Tracking.csv', 'G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Shoulder Trackingv4 TOP2.csv']

list_angle_smooth = []
list_angle_vel_rad = []
list_angle_acc_rad = []

window=11
poly=3

#extraction des csv
for i in range(len(files_path)):
    df = pd.read_csv(files_path[i], sep=';', decimal=',', skiprows=0)

    time = df.iloc[:, 0].to_numpy()
    dt = np.mean(np.diff(time))

    ox, oy = df.iloc[:, 7], df.iloc[:, 8]
    ax, ay = df.iloc[:, 9], df.iloc[:, 10]
    bx, by = df.iloc[:, 11], df.iloc[:, 12]

    oa_x, oa_y = ax - ox, ay - oy
    ob_x, ob_y = bx - ox, by - oy

    dot_product = oa_x * ob_x + oa_y * ob_y
    
    det = oa_x * ob_y - oa_y * ob_x

    angle_rad = np.arctan2(det, dot_product)
    angle_unwrapped = np.unwrap(angle_rad)

    smooth = savgol_filter(angle_unwrapped, window_length=window, polyorder=poly, deriv=0)
    vel = savgol_filter(angle_unwrapped, window_length=window, polyorder=poly, deriv=1, delta=dt)
    acc = savgol_filter(angle_unwrapped, window_length=window, polyorder=poly, deriv=2, delta=dt)

    list_angle_smooth.append(pd.Series(smooth))
    list_angle_vel_rad.append(pd.Series(vel))
    list_angle_acc_rad.append(pd.Series(acc))

All_angle_smooth = pd.concat(list_angle_smooth, axis=1)
All_angle_vel = pd.concat(list_angle_vel_rad, axis=1)
All_angle_acc = pd.concat(list_angle_acc_rad, axis=1)

def dir_vec(theta):
    return np.column_stack((np.cos(theta), np.sin(theta)))

# 1. Angles absolus (par rapport à la verticale)
a1 = All_angle_smooth.iloc[:, 0].to_numpy()
a2 = All_angle_smooth.iloc[:, 1].to_numpy()
a3 = All_angle_smooth.iloc[:, 2].to_numpy()

theta1 = -a1 + np.pi / 2
theta2 = theta1 - np.pi + a2
theta3 = np.pi - (-theta2 + a3)

# 2. Vitesses angulaires absolues (dtheta)
da1 = All_angle_vel.iloc[:, 0].to_numpy()
da2 = All_angle_vel.iloc[:, 1].to_numpy()
da3 = All_angle_vel.iloc[:, 2].to_numpy()

dtheta1 = -da1
dtheta2 = dtheta1 + da2
dtheta3 = dtheta2 - da3

# 3. Accélérations angulaires absolues (ddtheta)
dda1 = All_angle_acc.iloc[:, 0].to_numpy()
dda2 = All_angle_acc.iloc[:, 1].to_numpy()
dda3 = All_angle_acc.iloc[:, 2].to_numpy()

ddtheta1 = -dda1
ddtheta2 = ddtheta1 + dda2
ddtheta3 = ddtheta2 - dda3

n = len(theta1)

#calcul des positions
P_hand = np.zeros((n, 2))
P_elbow = P_hand + L1 * np.column_stack((np.sin(theta1), np.cos(theta1)))
P_shoulder = P_elbow + L2 * np.column_stack((np.sin(theta2), np.cos(theta2)))
P_pelvis = P_shoulder + L3 * np.column_stack((np.sin(theta3), np.cos(theta3)))
G_bust= P_shoulder + d3 * np.column_stack((np.sin(theta3), np.cos(theta3)))
G_lower = P_pelvis

def torque_calc_bottom_up(m_bust, m_lower, m_arm, m_forearm, L1, L2, L3, d3, theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, ddtheta1, ddtheta2, ddtheta3):
    
    # 1. Accélérations des centres de masse
    acc_elbow_x = L1 * (ddtheta1 * np.cos(theta1) - dtheta1**2 * np.sin(theta1))
    acc_elbow_y = - L1 * (ddtheta1 * np.sin(theta1) + dtheta1**2 * np.cos(theta1))

    acc_g1_x = r1 * L1 * (ddtheta1 * np.cos(theta1) - dtheta1**2 * np.sin(theta1))
    acc_g1_y = - r1 * L1 * (ddtheta1 * np.sin(theta1) + dtheta1**2 * np.cos(theta1))

    acc_shoulder_x = acc_elbow_x + L2 * (ddtheta2 * np.cos(theta2) - dtheta2**2 * np.sin(theta2))
    acc_shoulder_y = acc_elbow_y - L2 * (ddtheta2 * np.sin(theta2) + dtheta2**2 * np.cos(theta2))

    acc_g2_x = acc_elbow_x + r2 * L2 * (ddtheta2 * np.cos(theta2) - dtheta2**2 * np.sin(theta2))
    acc_g2_y = acc_elbow_y - r2 * L2 * (ddtheta2 * np.sin(theta2) + dtheta2**2 * np.cos(theta2))

    acc_bust_x = acc_shoulder_x + d3 * (ddtheta3 * np.cos(theta3) - dtheta3**2 * np.sin(theta3))
    acc_bust_y = acc_shoulder_y - d3 * (ddtheta3 * np.sin(theta3) + dtheta3**2 * np.cos(theta3))

    acc_pelvis_x = acc_shoulder_x + L3 * (ddtheta3 * np.cos(theta3) - dtheta3**2 * np.sin(theta3))
    acc_pelvis_y = acc_shoulder_y - L3 * (ddtheta3 * np.sin(theta3) + dtheta3**2 * np.cos(theta3))

    # 2. Forces de Réaction aux barres (GRF) estimées
    M_tot = m_forearm + m_arm + m_bust + m_lower
    
    F_grf_x = (m_forearm * acc_g1_x + m_arm * acc_g2_x + m_bust * acc_bust_x + m_lower * acc_pelvis_x)
    F_grf_y = (m_forearm * acc_g1_y + m_arm * acc_g2_y + m_bust * acc_bust_y + m_lower * acc_pelvis_y) + M_tot * g

    # 3. Calcul de la Dynamique Inverse : Bottom-Up
    
    # --- ARTICULATION DE LA MAIN ---
    d_grip_x = 0.04 # 3 centimètres (à ajuster selon ta prise)
    d_grip_y = 0.00 
    
    # Le vrai couple à la main sert à contrer le moment créé par la force de réaction sur ce bras de levier
    tau_hand = (d_grip_x * F_grf_y) - (d_grip_y * F_grf_x)

    # --- SEGMENT AVANT-BRAS ---
    F_elbow_x = m_forearm * acc_g1_x - F_grf_x
    F_elbow_y = m_forearm * acc_g1_y - F_grf_y + m_forearm * g

    # Bras de levier (depuis le centre de masse G1)
    rx_G1_hand = - r1 * L1 * np.sin(theta1)
    ry_G1_hand = - r1 * L1 * np.cos(theta1)
    rx_G1_elbow = (1 - r1) * L1 * np.sin(theta1)
    ry_G1_elbow = (1 - r1) * L1 * np.cos(theta1)

    M_grf_on_forearm = rx_G1_hand * F_grf_y - ry_G1_hand * F_grf_x
    M_elbow_on_forearm = rx_G1_elbow * F_elbow_y - ry_G1_elbow * F_elbow_x

    tau_elbow = I_cm_forearm * ddtheta1 - M_grf_on_forearm - M_elbow_on_forearm

    # --- SEGMENT BRAS ---
    F_shoulder_x = m_arm * acc_g2_x - (-F_elbow_x)
    F_shoulder_y = m_arm * acc_g2_y - (-F_elbow_y) + m_arm * g

    rx_G2_elbow = - r2 * L2 * np.sin(theta2)
    ry_G2_elbow = - r2 * L2 * np.cos(theta2)
    rx_G2_shoulder = (1 - r2) * L2 * np.sin(theta2)
    ry_G2_shoulder = (1 - r2) * L2 * np.cos(theta2)

    M_elbow_on_arm = rx_G2_elbow * (-F_elbow_y) - ry_G2_elbow * (-F_elbow_x)
    M_shoulder_on_arm = rx_G2_shoulder * F_shoulder_y - ry_G2_shoulder * F_shoulder_x

    tau_shoulder = I_cm_arm * ddtheta2 - (-tau_elbow) - M_elbow_on_arm - M_shoulder_on_arm

    return tau_hand, tau_elbow, tau_shoulder

# N'oublie pas de mettre à jour ton appel de fonction :
tau_hand, tau_elbow, tau_shoulder = torque_calc_bottom_up(m_bust, m_lower, m_arm, m_forearm, L1, L2, L3, d3, theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, ddtheta1, ddtheta2, ddtheta3)

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
    x = [P_hand[i, 0], P_elbow[i, 0], P_shoulder[i, 0], P_pelvis[i, 0], P_pelvis[i, 0]]
    y = [P_hand[i, 1], P_elbow[i, 1], P_shoulder[i, 1], P_pelvis[i, 1], P_pelvis[i, 1] - 0.6]
    ligne_corps.set_data(x, y)

    trajectoire_coude.set_data(P_elbow[: i + 1, 0], P_elbow[: i + 1, 1])
    trajectoire_epaule.set_data(P_shoulder[: i + 1, 0], P_shoulder[: i + 1, 1])
    trajectoire_buste.set_data(P_pelvis[: i + 1, 0], P_pelvis[: i + 1, 1])

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