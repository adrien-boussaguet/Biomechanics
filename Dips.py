import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter

#Masses
BODY_MASS = 80  # kg
ADDED_WEIGHT_MASS = 60  # kg

forearm_mass = 2 * 0.016 * BODY_MASS
upper_arm_mass = 2 * 0.028 * BODY_MASS
legs_mass = 2 * 0.161 * BODY_MASS
trunk_mass = BODY_MASS - forearm_mass - upper_arm_mass - legs_mass
lower_body_mass = legs_mass + ADDED_WEIGHT_MASS
total_system_mass = BODY_MASS + ADDED_WEIGHT_MASS

GRAVITY = 9.81

#Segments
FOREARM_LENGTH = 0.27  # m - avant-bras
UPPER_ARM_LENGTH = 0.31  # m - bras
TRUNK_LENGTH = 0.45  # m - buste
TRUNK_COM_DISTANCE = 0.30  # m - centre de masse corps

FOREARM_COM_RATIO = 0.430
FOREARM_GYRATION_RATIO = 0.303
UPPER_ARM_COM_RATIO = 0.436
UPPER_ARM_GYRATION_RATIO = 0.322

forearm_com_ratio_distal = 1 - FOREARM_COM_RATIO
upper_arm_com_ratio_distal = 1 - UPPER_ARM_COM_RATIO

#Inertie et bras de levier
forearm_moment_of_inertia = forearm_mass * (FOREARM_GYRATION_RATIO * FOREARM_LENGTH) ** 2
upper_arm_moment_of_inertia = upper_arm_mass * (UPPER_ARM_GYRATION_RATIO * UPPER_ARM_LENGTH) ** 2
trunk_moment_of_inertia = 2.5  # pas utilise pour l'instant

elbow_lever_arm = 0.025  # inutilise pour l'instant
shoulder_lever_arm = 0.045

# --- Tension mecanique (contrainte normale de traction sigma = F_m * cos(alpha_p) / PCSA) ---
# PCSA (Physiological Cross-Sectional Area) et angle de pennation, cf. quantification_tension_mecanique.txt
TRICEPS_PCSA = 35e-4                        # m^2 (35 cm^2)
TRICEPS_PENNATION_ANGLE = np.radians(15)    # rad

PEC_PCSA = 28e-4                            # m^2 (28 cm^2)
PEC_PENNATION_ANGLE = np.radians(18)        # rad
PEC_TENDON_LEVER_ARM = 0.04                 # m - bras de levier tendon pectoral (r_pec)
PEC_FORCE_SHARE = 0.65                      # part du grand pectoral dans le complexe presseur anterieur a l'epaule

# --- Extraction CSV ---
csv_file_paths = [
    r'G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Hand Tracking.csv',
    r'G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Elbow Tracking.csv',
    r'G:\Documents\VS Code\Biomecha\Biomechanics\Tracking\Shoulder Trackingv4 TOP2.csv',
]

window = 11
poly = 3


def load_tracking_angle(csv_path):
    df = pd.read_csv(csv_path, sep=';', decimal=',', skiprows=0)

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

    return smooth, vel, acc, dt


def direction_vector(theta):
    return np.column_stack((np.cos(theta), np.sin(theta)))


def compute_joint_torques(
    forearm_mass, upper_arm_mass, trunk_mass, lower_body_mass, total_mass,
    forearm_length, upper_arm_length, trunk_length, trunk_com_distance,
    forearm_angle, upper_arm_angle, trunk_angle,
    forearm_angular_velocity, upper_arm_angular_velocity, trunk_angular_velocity,
    forearm_angular_acceleration, upper_arm_angular_acceleration, trunk_angular_acceleration,
):
    elbow_acceleration_x = forearm_length * (forearm_angular_acceleration * np.cos(forearm_angle) - forearm_angular_velocity**2 * np.sin(forearm_angle))
    elbow_acceleration_y = -forearm_length * (forearm_angular_acceleration * np.sin(forearm_angle) + forearm_angular_velocity**2 * np.cos(forearm_angle))

    shoulder_acceleration_x = elbow_acceleration_x + upper_arm_length * (upper_arm_angular_acceleration * np.cos(upper_arm_angle) - upper_arm_angular_velocity**2 * np.sin(upper_arm_angle))
    shoulder_acceleration_y = elbow_acceleration_y - upper_arm_length * (upper_arm_angular_acceleration * np.sin(upper_arm_angle) + upper_arm_angular_velocity**2 * np.cos(upper_arm_angle))

    forearm_com_acceleration_x = forearm_com_ratio_distal * forearm_length * (forearm_angular_acceleration * np.cos(forearm_angle) - forearm_angular_velocity**2 * np.sin(forearm_angle))
    forearm_com_acceleration_y = -forearm_com_ratio_distal * forearm_length * (forearm_angular_acceleration * np.sin(forearm_angle) + forearm_angular_velocity**2 * np.cos(forearm_angle))

    upper_arm_com_acceleration_x = elbow_acceleration_x + upper_arm_com_ratio_distal * upper_arm_length * (upper_arm_angular_acceleration * np.cos(upper_arm_angle) - upper_arm_angular_velocity**2 * np.sin(upper_arm_angle))
    upper_arm_com_acceleration_y = elbow_acceleration_y - upper_arm_com_ratio_distal * upper_arm_length * (upper_arm_angular_acceleration * np.sin(upper_arm_angle) + upper_arm_angular_velocity**2 * np.cos(upper_arm_angle))

    trunk_com_acceleration_x = shoulder_acceleration_x + trunk_com_distance * (trunk_angular_acceleration * np.cos(trunk_angle) - trunk_angular_velocity**2 * np.sin(trunk_angle))
    trunk_com_acceleration_y = shoulder_acceleration_y - trunk_com_distance * (trunk_angular_acceleration * np.sin(trunk_angle) + trunk_angular_velocity**2 * np.cos(trunk_angle))

    pelvis_acceleration_x = shoulder_acceleration_x + trunk_length * (trunk_angular_acceleration * np.cos(trunk_angle) - trunk_angular_velocity**2 * np.sin(trunk_angle))
    pelvis_acceleration_y = shoulder_acceleration_y - trunk_length * (trunk_angular_acceleration * np.sin(trunk_angle) + trunk_angular_velocity**2 * np.cos(trunk_angle))

    # force de reaction transmise par les mains (= le "sol" du dip)
    hand_reaction_force_x = forearm_mass * forearm_com_acceleration_x + upper_arm_mass * upper_arm_com_acceleration_x + trunk_mass * trunk_com_acceleration_x + lower_body_mass * pelvis_acceleration_x
    hand_reaction_force_y = forearm_mass * forearm_com_acceleration_y + upper_arm_mass * upper_arm_com_acceleration_y + trunk_mass * trunk_com_acceleration_y + lower_body_mass * pelvis_acceleration_y + total_mass * GRAVITY

    # main
    grip_offset_x = 0.04
    grip_offset_y = 0.00
    hand_torque = grip_offset_x * hand_reaction_force_y - grip_offset_y * hand_reaction_force_x

    # coude
    elbow_joint_force_x = forearm_mass * forearm_com_acceleration_x - hand_reaction_force_x
    elbow_joint_force_y = forearm_mass * forearm_com_acceleration_y - hand_reaction_force_y + forearm_mass * GRAVITY

    forearm_com_to_hand_x = -forearm_com_ratio_distal * forearm_length * np.sin(forearm_angle)
    forearm_com_to_hand_y = -forearm_com_ratio_distal * forearm_length * np.cos(forearm_angle)
    forearm_com_to_elbow_x = (1 - forearm_com_ratio_distal) * forearm_length * np.sin(forearm_angle)
    forearm_com_to_elbow_y = (1 - forearm_com_ratio_distal) * forearm_length * np.cos(forearm_angle)

    hand_force_moment_on_forearm = forearm_com_to_hand_x * hand_reaction_force_y - forearm_com_to_hand_y * hand_reaction_force_x
    elbow_force_moment_on_forearm = forearm_com_to_elbow_x * elbow_joint_force_y - forearm_com_to_elbow_y * elbow_joint_force_x

    elbow_torque = forearm_moment_of_inertia * forearm_angular_acceleration - hand_force_moment_on_forearm - elbow_force_moment_on_forearm

    # epaule
    shoulder_joint_force_x = upper_arm_mass * upper_arm_com_acceleration_x + elbow_joint_force_x
    shoulder_joint_force_y = upper_arm_mass * upper_arm_com_acceleration_y + elbow_joint_force_y + upper_arm_mass * GRAVITY

    upper_arm_com_to_elbow_x = -upper_arm_com_ratio_distal * upper_arm_length * np.sin(upper_arm_angle)
    upper_arm_com_to_elbow_y = -upper_arm_com_ratio_distal * upper_arm_length * np.cos(upper_arm_angle)
    upper_arm_com_to_shoulder_x = (1 - upper_arm_com_ratio_distal) * upper_arm_length * np.sin(upper_arm_angle)
    upper_arm_com_to_shoulder_y = (1 - upper_arm_com_ratio_distal) * upper_arm_length * np.cos(upper_arm_angle)

    elbow_force_moment_on_upper_arm = upper_arm_com_to_elbow_x * (-elbow_joint_force_y) - upper_arm_com_to_elbow_y * (-elbow_joint_force_x)
    shoulder_force_moment_on_upper_arm = upper_arm_com_to_shoulder_x * shoulder_joint_force_y - upper_arm_com_to_shoulder_y * shoulder_joint_force_x

    shoulder_torque = upper_arm_moment_of_inertia * upper_arm_angular_acceleration + elbow_torque - elbow_force_moment_on_upper_arm - shoulder_force_moment_on_upper_arm

    return hand_torque, elbow_torque, shoulder_torque


def triceps_lever_arm(elbow_angle):
    return 0.021 + 0.005 * np.sin(elbow_angle)

# ---------------------------------------------------------------------------

smoothed_list, vel_list, acc_list = [], [], []
dt = None
for path in csv_file_paths:
    smooth, vel, acc, dt = load_tracking_angle(path)  # dt du dernier fichier reutilise plus bas (meme framerate suppose)
    smoothed_list.append(pd.Series(smooth))
    vel_list.append(pd.Series(vel))
    acc_list.append(pd.Series(acc))

all_angle_smooth = pd.concat(smoothed_list, axis=1)
all_angle_vel = pd.concat(vel_list, axis=1)
all_angle_acc = pd.concat(acc_list, axis=1)

# 1. Angles absolus (par rapport a la verticale)
a1 = all_angle_smooth.iloc[:, 0].to_numpy()
a2 = all_angle_smooth.iloc[:, 1].to_numpy()
a3 = all_angle_smooth.iloc[:, 2].to_numpy()

forearm_angle = -a1 + np.pi / 2
upper_arm_angle = forearm_angle - np.pi + a2
trunk_angle = np.pi - (-upper_arm_angle + a3)

# 2. Vitesses angulaires absolues
da1 = all_angle_vel.iloc[:, 0].to_numpy()
da2 = all_angle_vel.iloc[:, 1].to_numpy()
da3 = all_angle_vel.iloc[:, 2].to_numpy()

forearm_angular_velocity = -da1
upper_arm_angular_velocity = forearm_angular_velocity + da2
trunk_angular_velocity = upper_arm_angular_velocity - da3

# 3. Accelerations angulaires absolues
dda1 = all_angle_acc.iloc[:, 0].to_numpy()
dda2 = all_angle_acc.iloc[:, 1].to_numpy()
dda3 = all_angle_acc.iloc[:, 2].to_numpy()

forearm_angular_acceleration = -dda1
upper_arm_angular_acceleration = forearm_angular_acceleration + dda2
trunk_angular_acceleration = upper_arm_angular_acceleration - dda3

n_frames = len(forearm_angle)

# positions
hand_position = np.zeros((n_frames, 2))
elbow_position = hand_position + FOREARM_LENGTH * np.column_stack((np.sin(forearm_angle), np.cos(forearm_angle)))
shoulder_position = elbow_position + UPPER_ARM_LENGTH * np.column_stack((np.sin(upper_arm_angle), np.cos(upper_arm_angle)))
pelvis_position = shoulder_position + TRUNK_LENGTH * np.column_stack((np.sin(trunk_angle), np.cos(trunk_angle)))
trunk_com_position = shoulder_position + TRUNK_COM_DISTANCE * np.column_stack((np.sin(trunk_angle), np.cos(trunk_angle)))
lower_body_com_position = pelvis_position

hand_torque, elbow_torque, shoulder_torque = compute_joint_torques(
    forearm_mass, upper_arm_mass, trunk_mass, lower_body_mass, total_system_mass,
    FOREARM_LENGTH, UPPER_ARM_LENGTH, TRUNK_LENGTH, TRUNK_COM_DISTANCE,
    forearm_angle, upper_arm_angle, trunk_angle,
    forearm_angular_velocity, upper_arm_angular_velocity, trunk_angular_velocity,
    forearm_angular_acceleration, upper_arm_angular_acceleration, trunk_angular_acceleration,
)

triceps_lever = triceps_lever_arm(a2)
triceps_force = np.abs(elbow_torque) / triceps_lever
triceps_impulse = np.sum(triceps_force * dt)

# --- Tension mecanique (contrainte) triceps : sigma(t) = F_triceps(t) * cos(alpha_p) / PCSA ---
triceps_stress = triceps_force * np.cos(TRICEPS_PENNATION_ANGLE) / TRICEPS_PCSA  # Pa
triceps_stress_impulse = np.sum(triceps_stress * dt)  # Pa.s -> J_rep,triceps

# --- Tension mecanique (contrainte) grand pectoral ---
# Force du complexe presseur anterieur a l'epaule = |couple epaule| / bras de levier tendineux,
# puis part attribuee au grand pectoral (~65% du complexe, cf. quantification_tension_mecanique.txt)
pec_complex_force = np.abs(shoulder_torque) / PEC_TENDON_LEVER_ARM
pec_force = PEC_FORCE_SHARE * pec_complex_force
pec_stress = pec_force * np.cos(PEC_PENNATION_ANGLE) / PEC_PCSA  # Pa
pec_stress_impulse = np.sum(pec_stress * dt)  # Pa.s -> J_rep,pec

print(f"Contrainte mecanique pic triceps    : {triceps_stress.max() / 1e3:.1f} kPa | J_rep = {triceps_stress_impulse / 1e6:.3f} MPa.s")
print(f"Contrainte mecanique pic pectoraux  : {pec_stress.max() / 1e3:.1f} kPa | J_rep = {pec_stress_impulse / 1e6:.3f} MPa.s")

frames_x = np.arange(n_frames)

# --- Plots ---
fig, (ax_anim, ax_torque, ax_tension) = plt.subplots(1, 3, figsize=(18, 5))

ax_anim.set_xlim(-0.5, 1)
ax_anim.set_ylim(-1, 1)
ax_anim.set_aspect("equal")
ax_anim.set_title("Cinematique du mouvement")
ax_anim.set_xlabel("X (m)")
ax_anim.set_ylabel("Y (m)")
ax_anim.grid(True, alpha=0.3)

(body_line,) = ax_anim.plot([], [], "ko-", lw=3, markersize=8, label="Segments")
(elbow_traj,) = ax_anim.plot([], [], "r--", lw=1, alpha=0.5, label="Coude")
(shoulder_traj,) = ax_anim.plot([], [], "g--", lw=1, alpha=0.5, label="Epaule")
(pelvis_traj,) = ax_anim.plot([], [], "b--", lw=1, alpha=0.5, label="Bassin")
ax_anim.legend(loc="upper right")

ax_torque.set_xlim(0, n_frames)
y_min = min(hand_torque.min(), elbow_torque.min(), shoulder_torque.min())
y_max = max(hand_torque.max(), elbow_torque.max(), shoulder_torque.max())
margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
ax_torque.set_ylim(y_min - margin, y_max + margin)

ax_torque.set_title("Couples articulaires instantanes")
ax_torque.set_xlabel("Frames / Temps")
ax_torque.set_ylabel(r"Torque ($\mathrm{N}\cdot\mathrm{m}$)")
ax_torque.grid(True, alpha=0.3)

(line_tau_hand,) = ax_torque.plot([], [], "m-", lw=2, label="Torque Main")
(line_tau_elbow,) = ax_torque.plot([], [], "r-", lw=2, label="Torque Coude")
(line_tau_shoulder,) = ax_torque.plot([], [], "g-", lw=2, label="Torque Epaule")
ax_torque.legend(loc="upper right")

ax_tension.set_xlim(0, n_frames)
stress_max_kpa = max(triceps_stress.max(), pec_stress.max()) / 1e3
ax_tension.set_ylim(0, stress_max_kpa * 1.1)
ax_tension.set_title("Tension mecanique (contrainte)")
ax_tension.set_xlabel("Frames / Temps")
ax_tension.set_ylabel(r"Contrainte $\sigma$ (kPa)")
ax_tension.grid(True, alpha=0.3)

(line_sigma_triceps,) = ax_tension.plot([], [], "b-", lw=2, label="Triceps")
(line_sigma_pec,) = ax_tension.plot([], [], color="darkorange", lw=2, label="Grand pectoral")
ax_tension.legend(loc="upper right")

plt.tight_layout()


def animate(i):
    x = [hand_position[i, 0], elbow_position[i, 0], shoulder_position[i, 0], pelvis_position[i, 0], pelvis_position[i, 0]]
    y = [hand_position[i, 1], elbow_position[i, 1], shoulder_position[i, 1], pelvis_position[i, 1], pelvis_position[i, 1] - 0.6]
    body_line.set_data(x, y)

    elbow_traj.set_data(elbow_position[: i + 1, 0], elbow_position[: i + 1, 1])
    shoulder_traj.set_data(shoulder_position[: i + 1, 0], shoulder_position[: i + 1, 1])
    pelvis_traj.set_data(pelvis_position[: i + 1, 0], pelvis_position[: i + 1, 1])

    t = frames_x[: i + 1]
    line_tau_hand.set_data(t, hand_torque[: i + 1])
    line_tau_elbow.set_data(t, elbow_torque[: i + 1])
    line_tau_shoulder.set_data(t, shoulder_torque[: i + 1])

    line_sigma_triceps.set_data(t, triceps_stress[: i + 1] / 1e3)
    line_sigma_pec.set_data(t, pec_stress[: i + 1] / 1e3)

    return (body_line, elbow_traj, shoulder_traj, pelvis_traj, line_tau_hand, line_tau_elbow, line_tau_shoulder, line_sigma_triceps, line_sigma_pec)


ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=30, blit=True)

plt.show()