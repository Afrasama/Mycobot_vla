import pybullet as p
import os

p.connect(p.DIRECT)

# Load the robot
urdf_path = "urdf/mycobot_320.urdf"
robot = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

print("=== ROBOT JOINT INFO ===")
num_joints = p.getNumJoints(robot)
print(f"Total Joints: {num_joints}")

active_joints = []
for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    joint_name = info[1].decode('utf-8')
    joint_type = info[2]
    joint_type_str = {
        p.JOINT_REVOLUTE: "REVOLUTE",
        p.JOINT_PRISMATIC: "PRISMATIC",
        p.JOINT_SPHERICAL: "SPHERICAL",
        p.JOINT_PLANAR: "PLANAR",
        p.JOINT_FIXED: "FIXED"
    }.get(joint_type, "UNKNOWN")
    
    print(f"Joint {i}: {joint_name} | Type: {joint_type_str} | Lower: {info[8]:.2f} | Upper: {info[9]:.2f}")
    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        active_joints.append(i)

print(f"\nActive (Revolute/Prismatic) Joints: {active_joints}")

# Test calculateInverseKinematics
ee_index = 6
target_pos = [0.15, 0.0, 0.20]
target_orn = p.getQuaternionFromEuler([0.0, 3.14159, 0.0])

ik_output = p.calculateInverseKinematics(robot, ee_index, target_pos, target_orn)
print(f"\nIK Output Length: {len(ik_output)}")
print(f"IK Output: {ik_output}")

p.disconnect()
