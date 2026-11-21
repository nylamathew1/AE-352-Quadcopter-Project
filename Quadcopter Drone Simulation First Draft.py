import numpy as np
import matplotlib.pyplot as plt

# Time and gravity
DT = 0.01
G = 9.81

# Component masses (kg)
FRAME_MASS = 0.224
BATTERY_MASS = 0.223
ESC_MASS = 0.113
FLIGHTC_MASS = 0.0081
MOTOR_MASS = 0.0467
PROP_MASS = 0.0064
MASS = FRAME_MASS + BATTERY_MASS + ESC_MASS + FLIGHTC_MASS + 4*MOTOR_MASS + 4*PROP_MASS

# Geometry and rotor locations
WHEELBASE = 0.2746
A = WHEELBASE / (2*np.sqrt(2))
ROTOR_POS = np.array([[A, -A, 0], [A, A, 0], [-A, A, 0], [-A, -A, 0]])

# Motor and propeller values
BATTERY_VOLTAGE = 22.2
BATTERY_CAPACITY_AH = 1.380
BATTERY_WH = BATTERY_VOLTAGE * BATTERY_CAPACITY_AH
MOTOR_KV = 1500.0
MAX_OMEGA = MOTOR_KV * BATTERY_VOLTAGE * 2*np.pi/60
MAX_THRUST = 2.15 * G
THRUST_COEFF = MAX_THRUST / MAX_OMEGA**2
PROP_INERTIA = 89.2e-7

YAW_TORQUE_RATIO = 0.012
MOTOR_RESPONSE_TIME = 0.06
EFFICIENCY = 5.5

SPIN_DIRECTIONS = np.array([1, -1, 1, -1])
KQ = YAW_TORQUE_RATIO * THRUST_COEFF

B = np.array([
    [1, 1, 1, 1],
    ROTOR_POS[:, 1],
    -ROTOR_POS[:, 0],
    -YAW_TORQUE_RATIO * SPIN_DIRECTIONS
])

MOTOR_RESISTANCE = 0.1  

MOTOR_KT = 60 / (2 * np.pi * MOTOR_KV)
MOTOR_KE = MOTOR_KT


def calculate_inertia():
    motor_mass = MOTOR_MASS + PROP_MASS
    Ixx = np.sum(motor_mass * ROTOR_POS[:, 1]**2)
    Iyy = np.sum(motor_mass * ROTOR_POS[:, 0]**2)
    Izz = np.sum(motor_mass * (ROTOR_POS[:, 0]**2 + ROTOR_POS[:, 1]**2))

    center_mass = FRAME_MASS + BATTERY_MASS + ESC_MASS + FLIGHTC_MASS
    radius = 0.06
    height = 0.04
    Ixx += center_mass * (3*radius**2 + height**2)/12
    Iyy += center_mass * (3*radius**2 + height**2)/12
    Izz += center_mass * radius**2/2
    return np.diag([Ixx, Iyy, Izz])


INERTIA = calculate_inertia()


def rotation_matrix(phi, theta, psi):
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [cy*ct, cy*st*sp-sy*cp, cy*st*cp+sy*sp],
        [sy*ct, sy*st*sp+cy*cp, sy*st*cp-cy*sp],
        [-st, ct*sp, ct*cp]
    ])


def eom(state, thrusts, rotor_accel):
    # State: x,y,z,roll,pitch,yaw,vx,vy,vz,p,q,r,omega1,...,omega4
    phi, theta, psi = state[3:6]
    vx, vy, vz = state[6:9]
    p, q, r = state[9:12]

    #ang velocity to roll, pitch, and yaw
    phi_dot = p + np.sin(phi)*np.tan(theta)*q + np.cos(phi)*np.tan(theta)*r
    theta_dot = np.cos(phi)*q - np.sin(phi)*r
    psi_dot = (np.sin(phi)*q + np.cos(phi)*r)/np.cos(theta)

    #total thrust 
    total_thrust = np.sum(thrusts)
    force = rotation_matrix(phi, theta, psi) @ np.array([0, 0, total_thrust])

    # Linear acceleration from thrust and gravity
    acceleration = force/MASS - np.array([0, 0, G])

    #roll, pitch, yaw from four rotors
    torques = B[1:4] @ thrusts
    torques[2] = -np.sum(SPIN_DIRECTIONS * (KQ*state[12:16]**2 + PROP_INERTIA*rotor_accel))
    omega = state[9:12]
    angular_accel = np.linalg.solve(INERTIA, torques - np.cross(omega, INERTIA @ omega))

    #return derivs of position, orientation, and ang velocity
    return np.array([
        vx, vy, vz, phi_dot, theta_dot, psi_dot,
        acceleration[0], acceleration[1], acceleration[2],
        angular_accel[0], angular_accel[1], angular_accel[2]
    ])


def controller(state, target, target_vel, desired_yaw):
    KP_pos = 1.5
    KD_pos = 3

    # Calculate position and velocity errors relative to target
    position_error = target - state[0:3]
    velocity_error = target_vel - state[6:9]
    desired_accel = KP_pos * position_error + KD_pos * velocity_error
   
    #roll and pitch for horizontal acceleration
    psi = state[5]
    desired_roll = (desired_accel[0]*np.sin(psi) - desired_accel[1]*np.cos(psi))/G
    desired_pitch = (desired_accel[0]*np.cos(psi) + desired_accel[1]*np.sin(psi))/G
    max_angle = np.deg2rad(25);

    if desired_roll > max_angle:
        desired_roll = max_angle
    elif desired_roll < -max_angle:
        desired_roll = -max_angle

    if desired_pitch > max_angle:
        desired_pitch = max_angle
    elif desired_pitch < -max_angle:
        desired_pitch = -max_angle

    # total thrust for vertical acceleration
    phi, theta = state[3:5]
    total_thrust = MASS*(G + desired_accel[2])
    total_thrust = total_thrust/(np.cos(phi)*np.cos(theta))

    KP_att = np.array([12.0, 12.0, 5.0])
    KD_att = np.array([7.0, 7.0, 3.0])

    desired_angles = np.array([desired_roll, desired_pitch, desired_yaw])
    angle_error = desired_angles - state[3:6]

    # Wrap yaw error
    angle_error[2] = (angle_error[2] + np.pi) % (2 * np.pi) - np.pi

    # Calculate desired angular acceleration
    desired_angular_accel = (
        KP_att * angle_error
        - KD_att * state[9:12]
    )

    #angular acceleration into moments
    desired_torques = INERTIA @ desired_angular_accel

    # total thrust and torques into individual motor thrusts
    commands = np.array([total_thrust, *desired_torques])
    motor_thrusts = np.linalg.solve(B, commands)

    for i in range(4):
        if motor_thrusts[i] < 0:
            motor_thrusts[i] = 0
        elif motor_thrusts[i] > MAX_THRUST:
            motor_thrusts[i] = MAX_THRUST

    return motor_thrusts


def mission_command(t, mission):
    # Mission 1
    if mission == 'hover':
        target = np.array([0, 0, 1])
        target_vel = np.zeros(3)
        desired_yaw = 0

    # Mission 2
    if mission == 'circle':
        if t < 10:
            # Take off to 1 m
            target = np.array([0, 0, 1])
            target_vel = np.zeros(3)
        else:
            # Circular flight with radius 2 m and speed 0.5 m/s
            radius = 2.0
            speed = 0.5
            angular_speed = speed / radius
            angle = angular_speed * (t - 10)

            # Desired position
            x = radius * np.cos(angle) - radius
            y = radius * np.sin(angle)
            z = 1

            target = np.array([x, y, z])

            # Desired velocity
            vx = -speed * np.sin(angle)
            vy = speed * np.cos(angle)
            vz = 0

            target_vel = np.array([vx, vy, vz])

        desired_yaw = 0

    # Mission 3
    if mission == 'maneuver':
        target_vel = np.zeros(3)

        if t < 10:

            target = np.array([0, 0, 1])
            desired_yaw = 0

        elif t < 15:
            target = np.array([t - 10, 0, 1])
            target_vel = np.array([1, 0, 0])
            desired_yaw = 0        
            
        elif t < 30:
            target = np.array([5, 0, 1])
            desired_yaw = 0

        elif t < 40:
            target = np.array([5, 0, 1])
            desired_yaw = np.pi/2

        elif t < 45:
            target = np.array([5, t - 40, 1])
            target_vel = np.array([0, 1, 0])
            desired_yaw = np.pi / 2

        elif t < 60:
            target = np.array([5, 5, 1])
            desired_yaw = np.pi/2

        elif t < 185:
            z = 1 - 0.008 * (t - 60)
            target = np.array([5, 5, z])
            target_vel = np.array([0, 0, -0.008])
            desired_yaw = np.pi / 2

        else:
            target = np.array([5, 5, 0])
            target_vel = np.zeros(3)
            desired_yaw = np.pi/2

    return target, target_vel, desired_yaw



def run_simulation(mission, duration):
    # Initialize
    state = np.zeros(16)
    hover_thrust = MASS * G / 4
    hover_speed = np.sqrt(hover_thrust / THRUST_COEFF)
    state[12:16] = hover_speed
    times = np.arange(0, duration + DT/2, DT)
    states = np.zeros((len(times), 16))
    battery_wh = BATTERY_WH

    # Run simulation at each time step
    for i in range(len(times)):
        t = times[i]

        # Get target position, velocity, and yaw for the mission
        target, target_vel, desired_yaw = mission_command(t, mission)

        # Controller calculates desired thrust for each motor
        desired_thrusts = controller(state, target, target_vel, desired_yaw)

        # Calculate actual thrust based on current rotor speeds
        rotor_speed = state[12:16]
        thrusts = THRUST_COEFF * rotor_speed**2

        # Store current state for plotting
        states[i] = state

        # desired rotor speeds from motor thrust commands
        desired_speed = np.sqrt(np.maximum(desired_thrusts, 0)/THRUST_COEFF)

        # aerodynamic torque and motor torque
        aerodynamic_torque = KQ*rotor_speed**2
        motor_torque = PROP_INERTIA*(desired_speed - rotor_speed)/MOTOR_RESPONSE_TIME + aerodynamic_torque

        # Calculate motor current from motor torque
        motor_current = motor_torque / MOTOR_KT

        # Calculate motor voltage
        motor_voltage = (motor_current * MOTOR_RESISTANCE + MOTOR_KE * rotor_speed)

        # Calculate throttle
        motor_throttle = motor_voltage / BATTERY_VOLTAGE
        for i in range(4):
            if motor_throttle[i] < 0:
                motor_throttle[i] = 0
            elif motor_throttle[i] > 1:
                motor_throttle[i] = 1


        # rotor angular acceleration
        rotor_accel = (motor_torque - aerodynamic_torque)/PROP_INERTIA

        # derivatives using equations of motion
        state_dot = np.concatenate((eom(state, thrusts, rotor_accel), rotor_accel))

        # Updated drone position, orientation, velocities, and rotor speeds
        state = state + DT*state_dot
        total_thrusts = np.sum(thrusts);

        # Estimate battery power consumption and remaining energy
        power_w = total_thrusts * 1000 / (G * EFFICIENCY)        
        energy_used = power_w*DT/3600
        battery_wh = battery_wh - energy_used
        if battery_wh < 0:
            battery_wh = 0

    return times, states, battery_wh


def plot_results(times, states, mission):
    fig, axes = plt.subplots(3, 3, figsize=(12, 7))

    labels = ['x(m)', 'y(m)', 'z(m)', 'Roll(degrees)', 'Pitch(degrees)', 'Yaw(degrees)', 'Vx(m/s)', 'Vy(m/s)', 'Vz(m/s)']

    for row in range(3):
        for col in range(3):
            i = row *3 + col;

            if i < 3:
                values = states[:, i]
            elif i < 6:
                values = np.degrees(states[:, i])
            else:
                values = states[:, i]

            axes[row, col].plot(times, values)
            axes[row, col].set_title(labels[i])
            axes[row, col].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

# Mission 1
times, states, battery = run_simulation('hover', 135)

altitude = states[times >= 15, 2]

print('Average altitude:', round(np.mean(altitude), 3), 'm')
print('Battery remaining:', round(battery, 2), 'Wh')

plot_results(times, states, 'hover')


# Mission 2
times, states, battery = run_simulation('circle', 75)

x = states[times >= 15, 0]
y = states[times >= 15, 1]
vx = states[times >= 15, 6]
vy = states[times >= 15, 7]

radius = np.sqrt((x + 2)**2 + y**2)
speed = np.sqrt(vx**2 + vy**2)

print('Average radius:', round(np.mean(radius), 3), 'm')
print('Average speed:', round(np.mean(speed), 3), 'm/s')
print('Battery remaining:', round(battery, 2), 'Wh')

plot_results(times, states, 'circle')


# Mission 3
times, states, battery = run_simulation('maneuver', 185)

# Average velocity during 5 meters
first_speed = abs(states[int(15 / DT), 0] - states[int(10 / DT), 0]) / 5
second_speed = abs(states[int(45 / DT), 1] - states[int(40 / DT), 1]) / 5

# Landing speed
landing = (times >= 175) & (times < 185)
landing_speed = np.mean(np.abs(states[landing, 8]))

print('First movement average speed:', round(first_speed, 3), 'm/s')
print('Second movement average speed:', round(second_speed, 3), 'm/s')
print('Final descent average speed:', round(landing_speed, 3), 'm/s')
print('Final position:', np.round(states[-1, 0:3], 3), 'm')
print('Final yaw:', round(np.degrees(states[-1, 5]), 2), 'degrees')
print('Battery remaining:', round(battery, 2), 'Wh')

plot_results(times, states, 'maneuver')
