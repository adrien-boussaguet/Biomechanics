import numpy as np
import matplotlib.pyplot as plt

#Masses
m_body = 80 #kg
m_weight = 80 #kg
M = m_body + m_weight
g=9.81

#Segments
L1=0.272 #m - avant-bras
L2=0.346 #m - bras

theta1 = np.radians(-5) # angle of the forearm with respect to the horizontal
theta2 = np.radians(100) # angle of the upper arm with respect to the horizontal

P_hand=np.array([0,0])
P_elbow=L1*np.array([np.cos(theta1), np.sin(theta1)])
P_shoulder=P_elbow + L2*np.array([np.cos(theta2), np.sin(theta2)])

print("Hand Position:", P_hand)
print("Elbow Position:", P_elbow)
print("Shoulder Position:", P_shoulder)

y=[P_hand[0], P_elbow[0], P_shoulder[0]]
x=[P_hand[1], P_elbow[1], P_shoulder[1]]

plt.plot(x, y, 'o-')
plt.show()