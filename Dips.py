import numpy as np
import matplotlib.pyplot as plt

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

theta1 = np.radians(-5)  # angle of the forearm with respect to the vertical
theta2 = np.radians(100) # angle of the upper arm with respect to the vertical
theta3 = np.radians(180) # angle of the torso with respect to the vertical

#Position des articulations
P_hand=np.array([0,0])
P_elbow=L1*np.array([np.cos(theta1), np.sin(theta1)])
P_shoulder=P_elbow + L2*np.array([np.cos(theta2), np.sin(theta2)])
G_bust=P_shoulder+d3*np.array([np.cos(theta3), np.sin(theta3)])

#Fonction calcul de moment

def torque_elbow(theta1, theta2, theta3, alpha2):
    P_elbow=L1*np.array([np.cos(theta1), np.sin(theta1)])
    P_shoulder=P_elbow + L2*np.array([np.cos(theta2), np.sin(theta2)])
    G_bust=P_shoulder+d3*np.array([np.cos(theta3), np.sin(theta3)])

    gravity_lever_arm = G_bust[1] - P_elbow[1]
    tau_ext_elbow = - (m_bust * g * gravity_lever_arm)

    tau_int_elbow = I_elbow * alpha2 - tau_ext_elbow
    triceps_force = tau_int_elbow / la_elbow

    return tau_ext_elbow, tau_int_elbow, triceps_force

alpha2=1
tau_ext_elbow, tau_int_elbow, triceps_force = torque_elbow(theta1, theta2, theta3, alpha2)
#print(tau_ext_elbow, tau_int_elbow, triceps_force)

def torque_shoulder(theta1, theta2, theta3, alpha3):
    P_elbow=L1*np.array([np.cos(theta1), np.sin(theta1)])
    P_shoulder=P_elbow + L2*np.array([np.cos(theta2), np.sin(theta2)])
    G_bust=P_shoulder+d3*np.array([np.cos(theta3), np.sin(theta3)])

    gravity_lever_arm = G_bust[1] - P_shoulder[1]
    tau_ext_shoulder = - (m_bust * g * gravity_lever_arm)

    tau_int_shoulder = I_shoulder * alpha3 - tau_ext_shoulder
    pec_force = tau_int_shoulder / la_shoulder

    return tau_ext_shoulder, tau_int_shoulder, pec_force

alpha3=1
tau_ext_shoulder, tau_int_shoulder, pec_force = torque_shoulder(theta1, theta2, theta3, alpha3)
#print(tau_ext_shoulder, tau_int_shoulder, pec_force)

time = np.linspace(0, 3.0, 100)
theta2_high = np.radians(10) #Position haute de l'angle du coude
theta2_low = np.radians(100) #position basse de l'angle du coude
theta2_traj = theta2_low + (theta2_high - theta2_low) * (1 + np.cos(2 * np.pi * time / 3.0)) / 2



#Graphs
x=[P_hand[1], P_elbow[1], P_shoulder[1], G_bust[1]]
y=[P_hand[0], P_elbow[0], P_shoulder[0], G_bust[0]]

plt.figure(figsize=(6, 6))
plt.plot(x, y, 'o-')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlim(-0.75, 0.75)
plt.ylim(-0.75, 0.75)
plt.show()