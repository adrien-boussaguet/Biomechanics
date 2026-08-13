import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter

#Masses
m_body = 80 #kg
m_weight = 80 #kg
M = m_body + m_weight

m_forearm = 2 * 0.016 * m_body
m_arm = 2 * 0.028 * m_body
m_bust = M - m_forearm - m_arm

g=9.81

#Segments
L1=0.272 #m - avant-bras
L2=0.31 #m - bras
d3=0.60   #m - centre de masse corps

#Inertie et bras de levier
I_elbow = 2.5       #kg.m^2
la_elbow = 0.025    #m
I_shoulder = 42.5
la_shoulder = 0.045

#Extraction CSV
files_path = ['G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Hand Tracking.csv', 'G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Elbow Tracking.csv', 'G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Shoulder Trackingv4 TOP2.csv']

list_angle_rad = []

#effet de lissage
def apply_savgol(col):
    return savgol_filter(col, window_length=11, polyorder=3)

#extraction des csv
for i in range(len(files_path)):
    df = pd.read_csv(files_path[i], sep=';', decimal=',', skiprows=0)

    time = df.iloc[:, 0]

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
All_angle_smooth = All_angle_rad.apply(apply_savgol, axis=0)
print(All_angle_smooth)

def dir_vec(theta):
    return np.column_stack((np.cos(theta), np.sin(theta)))

#calcul des angles relatif à la vertical
theta1 = - All_angle_smooth.iloc[:, 0] + np.pi/2
theta2 = + theta1 - np.pi + All_angle_smooth.iloc[:, 1]
theta3 = np.pi - (- theta2 + All_angle_smooth.iloc[:, 2])

n = len(theta1)

#calcul des positions
P_hand = np.zeros((n, 2))
P_elbow = P_hand + L1 * np.column_stack((np.cos(theta1), np.sin(theta1)))
P_shoulder = P_elbow + L2 * np.column_stack((np.cos(theta2), np.sin(theta2)))
G_bust = P_shoulder + d3 * np.column_stack((np.cos(theta3), np.sin(theta3)))

#Graph
fig, ax = plt.subplots()
ax.set_xlim(-0.5, 1.5)  
ax.set_ylim(-1, 1)  
ax.set_aspect('equal')

ligne_corps, = ax.plot([], [], 'ko-', lw=3, markersize=8, label="Segments") 
trajectoire_coude, = ax.plot([], [], 'r--', lw=1, alpha=0.5, label="Coude")
trajectoire_epaule, = ax.plot([], [], 'g--', lw=1, alpha=0.5, label="Épaule")
trajectoire_buste, = ax.plot([], [], 'b--', lw=1, alpha=0.5, label="Buste")

ax.legend()

def animate(i):
    x = [P_hand[i, 1], P_elbow[i, 1], P_shoulder[i, 1], G_bust[i, 1]]
    y = [P_hand[i, 0], P_elbow[i, 0], P_shoulder[i, 0], G_bust[i, 0]]
    
    ligne_corps.set_data(x, y)
    
    trajectoire_coude.set_data(P_elbow[:i+1, 1], P_elbow[:i+1, 0])
    
    trajectoire_epaule.set_data(P_shoulder[:i+1, 1], P_shoulder[:i+1, 0])
    
    trajectoire_buste.set_data(G_bust[:i+1, 1], G_bust[:i+1, 0])
    
    return ligne_corps, trajectoire_coude, trajectoire_epaule, trajectoire_buste

ani = animation.FuncAnimation(fig, animate, frames=len(theta2), interval=30, blit=True)

plt.show()