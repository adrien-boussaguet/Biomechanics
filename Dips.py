import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
L2=0.346 #m - bras
d3=0.536   #m - centre de masse corps

#Inertie et bras de levier
I_elbow = 2.5       #kg.m^2
la_elbow = 0.025    #m
I_shoulder = 42.5
la_shoulder = 0.045

#Extraction CSV
#,'G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Shoulder Tracking.csv'
files_path = ['G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Elbow Tracking.csv','G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Shoulder Tracking.csv']

list_angle_rad = []

for i in range(len(files_path)):
    df = pd.read_csv(files_path[i], sep=';', decimal=',', skiprows=0)

    time = df.iloc[:, 0]

    ox, oy = df.iloc[:, 7], df.iloc[:, 8]
    ax, ay = df.iloc[:, 9], df.iloc[:, 10]
    bx, by = df.iloc[:, 11], df.iloc[:, 12]

    oa_x, oa_y = ax - ox, ay - oy
    ob_x, ob_y = bx - ox, by - oy

    dot_product = oa_x * ob_x + oa_y * ob_y
    norm_oa = np.sqrt(oa_x**2 + oa_y**2)
    norm_ob = np.sqrt(ob_x**2 + ob_y**2)

    cos_theta = np.clip(dot_product / (norm_oa * norm_ob), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)

    list_angle_rad.append(angle_rad)

All_angle_rad = pd.concat(list_angle_rad, axis=1)
print(All_angle_rad)

#Position des articulations
P_hand=np.array([0,0])
P_elbow=L1*np.array([np.cos(theta1), np.sin(theta1)])
P_shoulder=P_elbow + L2*np.array([np.cos(theta2), np.sin(theta2)])
G_bust=P_shoulder+d3*np.array([np.cos(theta3), np.sin(theta3)])

#def torque_elbow(theta1, theta2, theta3, alpha2):
    #P_elbow=L1*np.array([np.cos(theta1), np.sin(theta1)])
    #P_shoulder=P_elbow + L2*np.array([np.cos(theta2), np.sin(theta2)])
    #G_bust=P_shoulder+d3*np.array([np.cos(theta3), np.sin(theta3)])

    #gravity_lever_arm = G_bust[1] - P_elbow[1]
    #tau_ext_elbow = - (m_bust * g * gravity_lever_arm)

    #tau_int_elbow = I_elbow * alpha2 - tau_ext_elbow
    #triceps_force = tau_int_elbow / la_elbow

    #return tau_ext_elbow, tau_int_elbow, triceps_force

#def torque_shoulder(theta1, theta2, theta3, alpha3):
    #P_elbow=L1*np.array([np.cos(theta1), np.sin(theta1)])
    #P_shoulder=P_elbow + L2*np.array([np.cos(theta2), np.sin(theta2)])
    #G_bust=P_shoulder+d3*np.array([np.cos(theta3), np.sin(theta3)])

    #gravity_lever_arm = G_bust[1] - P_shoulder[1]
    #tau_ext_shoulder = - (m_bust * g * gravity_lever_arm)

    #tau_int_shoulder = I_shoulder * alpha3 - tau_ext_shoulder
    #pec_force = tau_int_shoulder / la_shoulder

    #return tau_ext_shoulder, tau_int_shoulder, pec_force

#Graphs
x=[P_hand[1], P_elbow[1], P_shoulder[1], G_bust[1]]
y=[P_hand[0], P_elbow[0], P_shoulder[0], G_bust[0]]

def animate(i):
    P_hand = np.array([0, 0])
    P_elbow = L1 * np.array([np.cos(theta1[i]), np.sin(theta1[i])])
    P_shoulder = P_elbow + L2 * np.array([np.cos(theta2[i]), np.sin(theta2[i])])
    G_bust = P_shoulder + d3 * np.array([np.cos(theta3[i]), np.sin(theta3[i])])
    
    # Coordonnées de l'instant t (avec ton inversion des axes [1] pour x et [0] pour y)
    x = [P_hand[1], P_elbow[1], P_shoulder[1], G_bust[1]]
    y = [P_hand[0], P_elbow[0], P_shoulder[0], G_bust[0]]
    
    # Mise à jour du corps
    ligne_corps.set_data(x, y)
    
    # Mise à jour de l'historique et de la trajectoire du COUDE
    hist_x_coude.append(P_elbow[1])
    hist_y_coude.append(P_elbow[0])
    trajectoire_coude.set_data(hist_x_coude, hist_y_coude)
    
    # Mise à jour de l'historique et de la trajectoire de l'ÉPAULE
    hist_x_epaule.append(P_shoulder[1])
    hist_y_epaule.append(P_shoulder[0])
    trajectoire_epaule.set_data(hist_x_epaule, hist_y_epaule)
    
    # Mise à jour de l'historique et de la trajectoire du BUSTE
    hist_x_buste.append(G_bust[1])
    hist_y_buste.append(G_bust[0])
    trajectoire_buste.set_data(hist_x_buste, hist_y_buste)
    
    return ligne_corps, trajectoire_coude, trajectoire_epaule, trajectoire_buste

# Création de l'animation
ani = animation.FuncAnimation(fig, animate, frames=frames, interval=30, blit=True)

plt.show()