import os
import sys
import time

import matplotlib.pyplot as plt  # type: ignore``
import numpy as np  # type: ignore
import pybullet as p  # type: ignore
import pybullet_data  # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.segmentation import (  # type: ignore
    capture_overhead_task_frame,
    configure_overhead_camera,
    get_overhead_pixels_per_meter,
    get_relative_pixel_error_overhead_and_rgb,
    save_overhead_debug_frame,
    sync_debug_workspace_camera,
    world_xy_to_overhead_pixel_error,
)
from reflection.llm_reflection_agent import LLMReflectionAgent, apply_policy_updates  # type: ignore
from reflection.vla_recovery_agent import VLARecoveryAgent  # type: ignore
from utils.gui_status import gui_status  # type: ignore
import math
from utils.logger import setup_execution_logger, log_robot_state, log_llm_decision, log_policy_update, log_session_summary

# Optional offline CNN (extra scene hints). Default off — use Ollama for reflection.
USE_OFFLINE_VISION_CLASSIFIER = os.getenv("USE_OFFLINE_VISION_CLASSIFIER", "0") == "1"
offline_classifier = None

# LLM-driven reflection agent. The backend can be set with LLM_AGENT_BACKEND.
# Primary backend: "ollama" for local Llama models.
USE_LLM_AGENT = os.getenv("USE_LLM_AGENT", "1") == "1"
FORCE_REFLECTION = os.getenv("FORCE_REFLECTION", "0") == "1"
FORCED_REFLECTION_ATTEMPTS = int(os.getenv("FORCED_REFLECTION_ATTEMPTS", "1"))
# Run LLM reflection after every completed round (success or failure), not only on grasp/place failures.
REFLECT_EVERY_ROUND = os.getenv("REFLECT_EVERY_ROUND", "1") == "1"
# Run VLA closed-loop recovery on any staged-pick failure (not only after a prior grasp_failure retry).
VLA_ON_ANY_PICK_FAILURE = os.getenv("VLA_ON_ANY_PICK_FAILURE", "1") == "1"
# After a successful staged pick, open gripper once and run VLA so logs show the recovery loop (demo only).
VLA_DEMO_MODE = os.getenv("VLA_DEMO_MODE", "0") == "1"
# Stress the full stack: aim miss + no automatic physical re-align until the last in-round attempt.
PIPELINE_EVAL_MODE = os.getenv("PIPELINE_EVAL_MODE", "0") == "1"

# ---------------- CONNECT ----------------
# Setup logging
os.environ["ROBOT_SESSION_ID"] = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
logger, log_file = setup_execution_logger()
logger.info(f"Log file: {log_file}")

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

p.setPhysicsEngineParameter(numSolverIterations=150)
p.setPhysicsEngineParameter(fixedTimeStep=1 / 240)

# ---------------- PLANE ----------------
plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=1.5)

# ---------------- LOAD ROBOT ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

if USE_OFFLINE_VISION_CLASSIFIER:
    try:
        from perception.offline_vision_classifier import OfflineVisionClassifier

        offline_model_path = os.path.join(
            BASE_DIR, "models", "offline_vlm", "tinycnn_direction.pt"
        )
        offline_classifier = OfflineVisionClassifier(model_path=offline_model_path)
        print("Offline vision classifier loaded:", offline_model_path)
    except Exception as exc:
        offline_classifier = None
        print("Offline vision classifier disabled:", exc)

robot = p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE,
)

# ---------------- FIND END EFFECTOR ----------------
ee_index = None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot, i)[12].decode() == "link6":
        ee_index = i
        break

print("End effector index:", ee_index)

# ---------------- PARALLEL GRIPPER (env: MYCOBOT_GRIPPER_URDF, basename under urdf/) ----------------
_gripper_name = os.getenv("MYCOBOT_GRIPPER_URDF", "wide_parallel_gripper.urdf").strip()
if not _gripper_name.lower().endswith(".urdf"):
    _gripper_name += ".urdf"
gripper_urdf = os.path.join(BASE_DIR, "urdf", os.path.basename(_gripper_name))
if not os.path.isfile(gripper_urdf):
    _fallback = os.path.join(BASE_DIR, "urdf", "wide_parallel_gripper.urdf")
    print(f"Warning: gripper URDF not found ({gripper_urdf}), using {_fallback}")
    gripper_urdf = _fallback
GRIPPER_MODEL_NAME = os.path.basename(gripper_urdf)
gripper = p.loadURDF(gripper_urdf)
logger.info(f"Gripper model: {GRIPPER_MODEL_NAME} path={gripper_urdf}")

# Increase friction on gripper fingers for better grasping
for j in range(-1, p.getNumJoints(gripper)):
    p.changeDynamics(
        gripper,
        j,
        lateralFriction=2.5,
        spinningFriction=0.5,
        rollingFriction=0.1,
        restitution=0.0
    )

# ---------------- TCP/TOOL OFFSET CALIBRATION ----------------
# Proper gripper offset calibration for accurate positioning
GRIPPER_TCP_OFFSET = [0.0, 0.0, 0.06]  # 6cm offset from link6 to gripper center (reduced)
# Must match move_to_position: tool Z down so parallel fingers straddle the cube from above.
GRIPPER_ORIENTATION = p.getQuaternionFromEuler([0.0, math.pi, 0.0])

def get_gripper_tcp_position():
    """TCP in world frame: link6 pose + offset rotated into link6 frame."""
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos, ee_orn = ee_state[0], ee_state[1]
    offset_world = np.array(p.rotateVector(ee_orn, GRIPPER_TCP_OFFSET))
    tcp_pos = np.array(ee_pos) + offset_world
    return tcp_pos, ee_orn


def set_gripper_tcp_target(target_pos, orientation=None):
    """IK target for link6 so that TCP (offset in link frame) reaches target_pos in world."""
    if orientation is None:
        orientation = GRIPPER_ORIENTATION
    # Use the TARGET orientation to compute the exact offset required for link6
    offset_world = np.array(p.rotateVector(orientation, GRIPPER_TCP_OFFSET))
    compensated_target = np.array(target_pos) - offset_world
    return compensated_target, orientation

# Get gripper joint indices for standard parallel gripper
gripper_joints = []
gripper_motor_joint = None

for i in range(p.getNumJoints(gripper)):
    joint_info = p.getJointInfo(gripper, i)
    joint_name = joint_info[1].decode()
    print(f"Found gripper joint: {joint_name} at index {i}")
    
    # Standard parallel gripper joint names
    if "finger_joint" in joint_name:
        gripper_joints.append(i)
        if "left_finger" in joint_name or "right_finger" in joint_name:
            gripper_motor_joint = i

print(f"Standard parallel gripper joints: {gripper_joints}")
print(f"Standard parallel gripper motor joint: {gripper_motor_joint}")

GRIPPER_FINGER_OPEN = 0.025
if gripper_joints:
    _uppers = [float(p.getJointInfo(gripper, j)[9]) for j in gripper_joints]
    if _uppers and max(_uppers) > 1e-6:
        GRIPPER_FINGER_OPEN = max(_uppers)

# Track target gripper joint position globally to keep it controlled during movement
target_gripper_joint_pos = GRIPPER_FINGER_OPEN

# Attach standard parallel gripper to robot end effector with proper TCP calibration
def attach_standard_gripper():
    """Attach standard parallel gripper to robot end effector"""
    # Get end effector position and orientation
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos, ee_orn = ee_state[0], ee_state[1]
    
    # Teleport gripper exactly to end effector position before constraining
    gripper_pos, gripper_orn = ee_pos, ee_orn
    p.resetBasePositionAndOrientation(gripper, gripper_pos, gripper_orn)
    
    # Attach gripper base exactly at the end effector (no offset)
    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee_index,
        childBodyUniqueId=gripper,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=[0, 0, 0, 1]
    )
    
    # Set constraint parameters for stable attachment with high force
    p.changeConstraint(constraint_id, maxForce=100000)  # High force for secure attachment
    
    # Disable ALL collisions between robot and gripper to prevent physical conflicts / IK locks
    for r_link in range(-1, p.getNumJoints(robot)):
        for g_link in range(-1, p.getNumJoints(gripper)):
            p.setCollisionFilterPair(robot, gripper, r_link, g_link, 0)
            
    print(f"Standard parallel gripper attached with calibrated TCP offset {GRIPPER_TCP_OFFSET}")
    print(f"Constraint ID: {constraint_id}")
    return constraint_id

# Preset robot joints to a pointing-straight-down starting posture to avoid IK jumps and physical locks
def preset_robot_joints_to_home():
    initial_target_link6 = [0.15, 0.0, 0.20] # A safe, centered, pointing-straight-down coordinate
    lower_limits = [-2.93, -2.35, -2.53, -2.53, -2.93, -3.14]
    upper_limits = [2.93, 2.35, 2.53, 2.53, 2.93, 3.14]
    joint_ranges = [5.86, 4.70, 5.06, 5.06, 5.86, 6.28]
    rest_poses = [0.0, -0.8, -1.0, 0.0, 1.57, 0.0]
    
    # Calculate a valid home configuration where ee index points down
    home_angles = p.calculateInverseKinematics(
        robot,
        ee_index,
        initial_target_link6,
        p.getQuaternionFromEuler([0.0, math.pi, 0.0]),
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        jointDamping=[0.1]*6,
        maxNumIterations=500,
        residualThreshold=1e-8
    )
    
    # Reset robot joints to this sensible starting state
    for idx, q in enumerate(home_angles):
        p.resetJointState(robot, idx, q)
    print(f"Robot joints initialized to pointing-straight-down posture: {home_angles}")

preset_robot_joints_to_home()

# Attach gripper
attach_standard_gripper()
for _gi in range(p.getNumJoints(gripper)):
    p.changeDynamics(gripper, _gi, lateralFriction=2.5, rollingFriction=0.001)
p.changeDynamics(gripper, -1, lateralFriction=2.5, rollingFriction=0.001)

# Gripper control functions with improved alignment and smooth control
def open_gripper():
    """Gradually open parallel fingers to URDF upper limit to prevent snap collision."""
    global target_gripper_joint_pos
    target_gripper_joint_pos = GRIPPER_FINGER_OPEN
    print(f"Gripper opening gradually to {GRIPPER_FINGER_OPEN:.4f} m ...")
    
    # Read current gripper joint states to start from active position
    try:
        current_pos = float(p.getJointState(gripper, gripper_joints[0])[0])
    except Exception:
        current_pos = 0.0

    # Gradually open fingers in a loop!
    for step in range(12):
        target = current_pos + (GRIPPER_FINGER_OPEN - current_pos) * ((step + 1) / 12.0)
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=50,  # High opening force to physically overcome friction and load!
                positionGain=0.2,  # Strong tracking gains
                velocityGain=0.5
            )
        # Simulate gradual opening steps
        for _ in range(3):
            p.stepSimulation()
            time.sleep(1/240)
            
    print("Gripper opened")
    # Add final stabilization
    for _ in range(int(0.15 * 240)):
        p.stepSimulation()
        time.sleep(1/240)

def close_gripper():
    """Close the wide parallel gripper gradually with proper mechanical grasping"""
    global target_gripper_joint_pos
    target_gripper_joint_pos = 0.0
    print("Closing gripper gradually...")
    
    # Gradual closing for better grip control
    for step in range(10):
        target = GRIPPER_FINGER_OPEN * (1.0 - step / 9.0)
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=12,  # Smooth, gradual closing force
                positionGain=0.1,
                velocityGain=0.3
            )
        # Simulate gradual closing
        for _ in range(5):
            p.stepSimulation()
            time.sleep(1/240)
    
    # Final close with realistic holding force and feedback gains to avoid chattering/collapsing
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(
            gripper,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=20,  # Safe holding force preventing squishing chattering
            positionGain=0.25,
            velocityGain=0.5
        )
    print("Gripper closed - mechanical grasp")
    # Add stabilization delay after closing
    for _ in range(int(0.5 * 240)):
        p.stepSimulation()
        time.sleep(1/240)

# Open gripper initially
open_gripper()

# Function to read robot joint angles and update GUI
def update_robot_joint_angles():
    """Read robot joint angles and update GUI display"""
    try:
        joint_angles = {}
        for i in range(p.getNumJoints(robot)):
            joint_info = p.getJointInfo(robot, i)
            if joint_info[2] == p.JOINT_REVOLUTE:  # Only get revolute joints
                joint_name = joint_info[1].decode()
                joint_angle = p.getJointState(robot, i)[0]  # Get angle in radians
                joint_angles_deg = math.degrees(joint_angle)  # Convert to degrees
                joint_angles[joint_name] = joint_angles_deg
        
        # Update GUI with joint angles
        gui_status.update_joint_angles(joint_angles)
    except Exception as e:
        print(f"Error reading joint angles: {e}")

# Smooth joint angles display on PyBullet screen (clean text only)
joint_text_ids = {}  # Store text object IDs for each joint

def create_smooth_joint_display():
    """Create smooth joint angle text display (no lines)"""
    global joint_text_ids
    joint_text_ids = {}
    
    # Create text for each revolute joint
    for i in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            joint_name = joint_info[1].decode()
            
            # Get joint position
            joint_state = p.getLinkState(robot, i)
            joint_pos = joint_state[0]
            
            # Position text above joint for visibility
            text_position = [
                joint_pos[0],
                joint_pos[1], 
                joint_pos[2] + 0.05  # Above joint
            ]
            
            # Create clean text (no lines)
            text_id = p.addUserDebugText(
                text=f"{joint_name}: 0.0°",
                textPosition=text_position,
                textColorRGB=[1, 1, 1],  # White text
                textSize=0.8,  # Small, clean text
                lifeTime=0  # Persistent
            )
            
            joint_text_ids[joint_name] = text_id

def update_smooth_joint_display():
    """Update joint angle display smoothly (no lines)"""
    global joint_text_ids
    
    for i in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            joint_name = joint_info[1].decode()
            joint_angle = p.getJointState(robot, i)[0]
            joint_angle_deg = math.degrees(joint_angle)
            
            # Get current joint position
            joint_state = p.getLinkState(robot, i)
            joint_pos = joint_state[0]
            
            # Position text above joint
            text_position = [
                joint_pos[0],
                joint_pos[1], 
                joint_pos[2] + 0.05  # Above joint
            ]
            
            # Remove old text and create new one
            if joint_name in joint_text_ids:
                p.removeUserDebugItem(joint_text_ids[joint_name])
            
            # Create updated text (no lines)
            text_id = p.addUserDebugText(
                text=f"{joint_name}: {joint_angle_deg:5.1f}°",
                textPosition=text_position,
                textColorRGB=[1, 1, 1],  # White text
                textSize=0.8,  # Small, clean text
                lifeTime=0  # Persistent
            )
            
            joint_text_ids[joint_name] = text_id

# Show joint angles display when robot starts working
gui_status.show_joint_angles_display(True)

# Joint angles only in GUI window (no screen display, no parameters)

# Improved grasp detection with centering
def check_grasp_stability(cube_body_id):
    """Check if gripper has stable centered grasp"""
    contacts = p.getContactPoints(gripper, cube_body_id)
    if len(contacts) == 0:
        return False, 0, 0
    
    # Check for parallel face contacts (not corners)
    parallel_contacts = 0
    for contact in contacts:
        # Check if contact normal is mostly vertical (parallel faces)
        normal = contact[7]
        if abs(normal[2]) > 0.7:  # Mostly vertical contact
            parallel_contacts += 1
    
    return len(contacts) >= 2, len(contacts), parallel_contacts

# Improved grasp function with retry mechanism - NO MAGNETIC GRABBING
def improved_grasp_object(cube_body_id, max_retries=3):
    """Grasp the object with improved alignment and retry mechanism - MECHANICAL ONLY"""
    for retry in range(max_retries):
        if retry > 0:
            print(f"Retry attempt {retry + 1}/{max_retries}")
            # Reopen gripper and reposition slightly
            open_gripper()
            p.stepSimulation()
            # Slight reposition for retry
            cube_pos, _ = p.getBasePositionAndOrientation(cube_body_id)
            reposition_offset = [
                cube_pos[0] + np.random.uniform(-0.005, 0.005),
                cube_pos[1] + np.random.uniform(-0.005, 0.005),
                cube_pos[2]
            ]
            # Query active joint limits for high accuracy IK
            joint_indices = []
            lower_limits = []
            upper_limits = []
            joint_ranges = []
            for j in range(p.getNumJoints(robot)):
                joint_info = p.getJointInfo(robot, j)
                if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    joint_indices.append(j)
                    lower_limits.append(joint_info[8])
                    upper_limits.append(joint_info[9])
                    joint_ranges.append(joint_info[9] - joint_info[8])
            
            # Attract towards straight/sensible home rest pose to prevent backward joint folding/stretching
            home_rest_poses = [0.0, -0.8, -1.0, 0.0, 1.57, 0.0]
            joint_damping = [0.1] * len(joint_indices)

            # Move to repositioned location
            joint_angles = p.calculateInverseKinematics(
                robot,
                ee_index,
                reposition_offset,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=home_rest_poses,
                jointDamping=joint_damping,
                maxNumIterations=500,
                residualThreshold=1e-6
            )
            
            for j in range(p.getNumJoints(robot)):
                joint_info = p.getJointInfo(robot, j)
                if joint_info[2] == p.JOINT_REVOLUTE:
                    p.setJointMotorControl2(
                        robot,
                        j,
                        p.POSITION_CONTROL,
                        targetPosition=joint_angles[j],
                        force=400,
                        positionGain=0.04,  # Slow movement for retry
                        velocityGain=0.3
                    )
            
            for _ in range(20):
                p.stepSimulation()
        
        # Close gripper for MECHANICAL grasping only
        close_gripper()
        
        # Wait for mechanical contact to establish
        for i in range(60):  # More steps for mechanical contact
            p.stepSimulation()
        
        # Check for MECHANICAL grasp with parallel alignment
        grasp_stable = False
        contact_count = 0
        centered_contacts = 0
        
        for i in range(40):  # More steps for stable mechanical grasp
            p.stepSimulation()
            stable, total_contacts, parallel_contacts = check_grasp_stability(cube_body_id)
            if stable:
                contact_count += 1
                if parallel_contacts >= 2:  # Need parallel face contacts
                    centered_contacts += 1
                
                if centered_contacts >= 3:  # Need 3 centered contact checks for mechanical grasp
                    grasp_stable = True
                    break
        
        if grasp_stable:
            print(f"Object grasped MECHANICALLY with centered alignment! (Attempt {retry + 1})")
            return True
        else:
            print(f"Mechanical grasp failed on attempt {retry + 1}, contacts: {contact_count}, centered: {centered_contacts}")
    
    print("Failed to grasp object mechanically after all retries")
    return False

# Improved movement with slow mode
def move_to_position(target_position, orientation=None, slow_mode=False):
    """Move robot to target position using inverse kinematics with optional slow mode"""
    if orientation is None:
        orientation = p.getQuaternionFromEuler([0, np.pi, 0])  # Gripper facing down (direct above)
    
    # Query active joint limits for high accuracy IK
    joint_indices = []
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    for j in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, j)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_indices.append(j)
            lower_limits.append(joint_info[8])
            upper_limits.append(joint_info[9])
            joint_ranges.append(joint_info[9] - joint_info[8])
    
    # Dynamic base joint bias to target angle to prevent IK local minima in negative/perpendicular quadrants
    target_yaw = math.atan2(target_position[1], target_position[0])
    home_rest_poses = [target_yaw, -0.8, -1.0, 0.0, 1.57, 0.0]
    joint_damping = [0.1] * len(joint_indices)

    # Calculate inverse kinematics with higher accuracy
    joint_angles = p.calculateInverseKinematics(
        robot,
        ee_index,
        target_position,
        orientation,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=home_rest_poses,
        jointDamping=joint_damping,
        maxNumIterations=500,  # More iterations for better accuracy
        residualThreshold=1e-6  # Tighter threshold
    )
    
    # Apply joint controls with slow mode option and safe force/velocity limits
    position_gain = 0.05 if slow_mode else 0.12  # Slower movement for final approach
    velocity_gain = 0.25 if slow_mode else 0.5  # Slower velocity for final approach
    max_vel = 1.2 if slow_mode else 2.2
    
    for j in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, j)
        if joint_info[2] == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[j],
                force=80.0,  # Safe force limit for 1kg arm to avoid chatter
                positionGain=position_gain,
                velocityGain=velocity_gain,
                maxVelocity=max_vel
            )

# ---------------- INTELLIGENT CUBE PLACEMENT ----------------
# Smart cube positioning based on robot capabilities and task strategy
import math

def calculate_optimal_cube_position():
    """Calculate optimal cube position for autonomous robot operation"""
    
    # Define reachability constant
    REACHABLE_THRESHOLD = 0.30
    
    # Strategy: Place cube in position that tests different skills
    # while being optimally reachable and challenging
    
    # Define strategic positions within reachable workspace
    strategic_positions = [
        # Front-right area (most common working area)
        (0.20, 0.15),
        # Front-left area  
        (0.20, -0.15),
        # Right side
        (0.15, 0.20),
        # Left side
        (0.15, -0.20),
        # Front-center (easiest)
        (0.25, 0.0),
        # Diagonal positions (more challenging)
        (0.18, 0.18),
        (0.18, -0.18),
    ]
    
    # Select position based on session number for variety
    session_id = hash(time.strftime("%Y%m%d")) % len(strategic_positions)
    base_position = strategic_positions[session_id]
    
    # Add small intelligent variation for learning
    variation_x = np.random.uniform(-0.02, 0.02)  # Small 2cm variation
    variation_y = np.random.uniform(-0.02, 0.02)
    
    optimal_x = base_position[0] + variation_x
    optimal_y = base_position[1] + variation_y
    
    # Ensure strictly within conservative reachability bounds (min 0.20m to prevent folding collapses, max 0.29m)
    distance = math.sqrt(optimal_x**2 + optimal_y**2)
    if distance > 0.29:
        scale = 0.29 / distance
        optimal_x *= scale
        optimal_y *= scale
    elif distance < 0.20:
        scale = 0.20 / distance
        optimal_x *= scale
        optimal_y *= scale
        
    # Enforce base joint revolute limit safety clamp (avoid dead-zone near 180 degrees)
    angle = math.atan2(optimal_y, optimal_x)
    if abs(angle) > 2.79:
        angle = math.copysign(2.79, angle)
        distance = math.sqrt(optimal_x**2 + optimal_y**2)
        optimal_x = distance * math.cos(angle)
        optimal_y = distance * math.sin(angle)
    
    return optimal_x, optimal_y

# Calculate intelligent cube position
optimal_x, optimal_y = calculate_optimal_cube_position()
cube = p.loadURDF("cube_small.urdf", [optimal_x, optimal_y, 0.02])
print(f"Cube placed at optimal position: ({optimal_x:.3f}, {optimal_y:.3f}, 0.02)")
print(f"Strategic placement for autonomous learning and testing")

# Check if cube is reachable by robot
robot_base_pos = np.array([0, 0, 0])  # Robot base at origin
cube_pos = np.array([optimal_x, optimal_y, 0.02])
distance_from_robot = np.linalg.norm(cube_pos[:2] - robot_base_pos[:2])  # Only X,Y distance

# MyCobot 320 reachability constants
MAX_REACH = 0.32
REACHABLE_THRESHOLD = 0.30  # Conservative threshold for reachability
ROBOT_BASE_XY = np.array([0.0, 0.0])
MIN_REACH_DISTANCE_M = float(os.getenv("MIN_REACH_DISTANCE_M", "0.20"))
MAX_REACH_DISTANCE_M = float(os.getenv("MAX_REACH_DISTANCE_M", str(REACHABLE_THRESHOLD)))


class CubeOutOfReachError(RuntimeError):
    """The cube left the XY workspace the arm can reach."""


def workspace_xy_distance(xy):
    pos = np.asarray(xy, dtype=float)[:2]
    return float(np.linalg.norm(pos - ROBOT_BASE_XY))


def check_cube_in_workspace(xy, label="cube"):
    """
    Verify cube XY is inside the annular reach zone (min/max radial distance from base).
    Returns (ok, distance_m, error_message).
    """
    dist = workspace_xy_distance(xy)
    x, y = float(xy[0]), float(xy[1])
    if dist > MAX_REACH_DISTANCE_M:
        return (
            False,
            dist,
            (
                f"ERROR: {label} left reachable workspace — "
                f"position ({x:.3f}, {y:.3f}) is {dist:.3f} m from base "
                f"(max {MAX_REACH_DISTANCE_M:.3f} m)"
            ),
        )
    if dist < MIN_REACH_DISTANCE_M:
        return (
            False,
            dist,
            (
                f"ERROR: {label} too close to robot base — "
                f"position ({x:.3f}, {y:.3f}) is {dist:.3f} m from base "
                f"(min {MIN_REACH_DISTANCE_M:.3f} m)"
            ),
        )
    return True, dist, ""


def reseat_cube_in_workspace():
    """Teleport cube back into the reachable annulus after a knock-off."""
    global optimal_x, optimal_y
    optimal_x, optimal_y = calculate_optimal_cube_position()
    p.resetBasePositionAndOrientation(cube, [optimal_x, optimal_y, 0.02], [0, 0, 0, 1])
    p.resetBaseVelocity(cube, [0, 0, 0], [0, 0, 0])
    ok, dist, msg = check_cube_in_workspace([optimal_x, optimal_y])
    if not ok:
        raise CubeOutOfReachError(msg)
    logger.info(
        f"CUBE_RESEATED: xy=({optimal_x:.4f},{optimal_y:.4f}) dist={dist:.4f}m"
    )
    return optimal_x, optimal_y


def report_cube_out_of_reach(msg, xy, dist):
    """Cube left workspace — retry via reflection when attempts remain, else end round."""
    global last_failure_type, state, cube_reach_error_reported
    if cube_reach_error_reported:
        return
    last_failure_type = "cube_out_of_reach"
    cube_reach_error_reported = True
    print(msg)
    logger.error(
        f"CUBE_OUT_OF_REACH: xy=({float(xy[0]):.4f},{float(xy[1]):.4f}) "
        f"dist={dist:.4f} min={MIN_REACH_DISTANCE_M:.3f} max={MAX_REACH_DISTANCE_M:.3f}"
    )
    gui_status.update_status("Error", msg[:140])
    gui_status.display_status()
    if retry_count < max_retries:
        print(
            "Scheduling LLM reflection retry (cube out of reach / knocked off table).",
            flush=True,
        )
        state = "analyze"
    else:
        print("Max in-round retries reached — ending round (cube out of reach).", flush=True)
        state = "done"


cube_reach_error_reported = False

_ok_init, _dist_init, _msg_init = check_cube_in_workspace([optimal_x, optimal_y])
if not _ok_init:
    raise CubeOutOfReachError(_msg_init)

print(f"Optimal cube distance from robot: {distance_from_robot:.3f}m")
print(
    f"Reachable workspace: {MIN_REACH_DISTANCE_M:.2f}–{MAX_REACH_DISTANCE_M:.2f} m from base"
)

configure_overhead_camera(min_reach_m=MIN_REACH_DISTANCE_M, max_reach_m=MAX_REACH_DISTANCE_M)
logger.info(f"OPTIMAL PLACEMENT: Cube positioned at ({optimal_x:.3f}, {optimal_y:.3f}) - {distance_from_robot:.3f}m from robot")
p.changeDynamics(
    cube,
    -1,
    lateralFriction=2.5,
    linearDamping=0.35,
    angularDamping=0.35,
    restitution=0.0,
)

# ---------------- INTELLIGENT GOAL SELECTION ----------------
def calculate_optimal_goal_position():
    """Calculate optimal goal position based on cube location and robot strategy"""
    
    # Strategy: Place goal in a position that requires different movement patterns
    cube_angle = math.atan2(optimal_y, optimal_x)
    cube_distance = math.sqrt(optimal_x**2 + optimal_y**2)
    
    # Choose goal position that requires different robot movement
    # Options: opposite side, perpendicular, or same side but different distance
    strategies = [
        # Opposite side (180 degrees)
        lambda: (-optimal_x * 0.85, -optimal_y * 0.85),
        # Perpendicular (90 degrees)
        lambda: (-optimal_y * 0.8, optimal_x * 0.8),
        # Same side but different distance
        lambda: (optimal_x * 0.8, optimal_y * 0.8),
        # Diagonal from cube
        lambda: ((optimal_x + optimal_y) * 0.55, (optimal_y - optimal_x) * 0.55),
    ]
    
    # Select strategy based on cube position for variety
    strategy_index = int(abs(cube_angle) * 2) % len(strategies)
    goal_x, goal_y = strategies[strategy_index]()
    
    # Ensure goal is reachable and not too close to robot base.
    # Floor is 0.22 m (not 0.20) so physics settling after placement
    # never slips the cube below the MIN_REACH_DISTANCE_M boundary.
    GOAL_MIN_DIST = max(0.22, MIN_REACH_DISTANCE_M + 0.02)
    goal_distance = math.sqrt(goal_x**2 + goal_y**2)
    if goal_distance > 0.29:
        scale = 0.29 / goal_distance
        goal_x *= scale
        goal_y *= scale
    elif goal_distance < GOAL_MIN_DIST:
        scale = GOAL_MIN_DIST / max(goal_distance, 1e-6)
        goal_x *= scale
        goal_y *= scale
        
        # Enforce base joint revolute limit safety clamp (avoid dead-zone near 180 degrees)
    goal_angle = math.atan2(goal_y, goal_x)
    if abs(goal_angle) > 2.79:
        goal_angle = math.copysign(2.79, goal_angle)
        goal_distance = math.sqrt(goal_x**2 + goal_y**2)
        goal_x = goal_distance * math.cos(goal_angle)
        goal_y = goal_distance * math.sin(goal_angle)
    
    return goal_x, goal_y

# Calculate intelligent goal position
goal_x, goal_y = calculate_optimal_goal_position()
goal_position = np.array([goal_x, goal_y, 0.02])
print(f"Goal set at strategic position: ({goal_x:.3f}, {goal_y:.3f}, 0.02)")
print(f"Goal distance from robot: {math.sqrt(goal_x**2 + goal_y**2):.3f}m")
logger.info(f"STRATEGIC GOAL: Goal positioned at ({goal_x:.3f}, {goal_y:.3f}) for optimal learning")

# Create goal object for visualization
goal = p.createMultiBody(
    baseVisualShapeIndex=p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=[1, 0, 0, 1]
    ),
    basePosition=[goal_x, goal_y, 0.02]
)

# ---------------- OVERHEAD TASK CAMERA ----------------
OVERHEAD_BODY_IDS = {"cube": cube, "gripper": gripper, "goal": goal, "robot": robot}
OVERHEAD_DEBUG_DIR = os.path.join(BASE_DIR, "data", "overhead_live")
SAVE_OVERHEAD_FRAMES = os.getenv("SAVE_OVERHEAD_FRAMES", "1") == "1"
# GUI view (3/4 angle). Top-down is only used for vision snapshots / segmentation.
SYNC_DEBUG_WORKSPACE_CAMERA = os.getenv(
    "SYNC_DEBUG_WORKSPACE_CAMERA",
    os.getenv("SYNC_OVERHEAD_DEBUG_CAMERA", "1"),
) == "1"


def _save_overhead_snapshot(tag: str, verbose: bool = False):
    """Write annotated overhead frame so grasp/VLA actions are reviewable."""
    if not SAVE_OVERHEAD_FRAMES:
        return
    try:
        rgb, seg, report = capture_overhead_task_frame(
            OVERHEAD_BODY_IDS, verbose=verbose
        )
        path = os.path.join(OVERHEAD_DEBUG_DIR, f"{tag}.png")
        save_overhead_debug_frame(rgb, path, seg, OVERHEAD_BODY_IDS)
        latest = os.path.join(OVERHEAD_DEBUG_DIR, "latest.png")
        save_overhead_debug_frame(rgb, latest, seg, OVERHEAD_BODY_IDS)
        if verbose or not report.get("all_visible", True):
            print(f"[OVERHEAD] saved {path} all_visible={report.get('all_visible')}")
    except Exception as exc:
        print(f"[OVERHEAD] snapshot failed ({tag}): {exc}")


try:
    _rgb0, _seg0, _rep0 = capture_overhead_task_frame(OVERHEAD_BODY_IDS, verbose=True)
    print(
        f"[OVERHEAD] task camera ready — "
        f"{_rep0['image_size'][0]}x{_rep0['image_size'][1]}, "
        f"{_rep0['pixels_per_meter']:.1f} px/m, "
        f"all_visible={_rep0['all_visible']}"
    )
    _save_overhead_snapshot("startup", verbose=True)
except Exception as _cam_exc:
    print(f"[OVERHEAD] initial capture failed: {_cam_exc}")

# ---------------- CAMERA (DEBUG VIEW — 3/4 for watching joints) ----------------
if SYNC_DEBUG_WORKSPACE_CAMERA:
    sync_debug_workspace_camera()
    print(
        "[CAMERA] GUI: 3/4 workspace view (adjust DEBUG_CAMERA_YAW/PITCH/DISTANCE in .env). "
        "Vision: top-down task camera only."
    )
else:
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=45,
        cameraPitch=-35,
        cameraTargetPosition=[0.22, 0.0, 0.10],
    )

# ---------------- POLICY ----------------
DEFAULT_POLICY = {
    "approach_height": 0.12,
    "grasp_height": 0.0,
    "lift_height": 0.20,
    "release_delay": 60,
    "x_offset": 0.0,
    "y_offset": 0.0,
}
policy = dict(DEFAULT_POLICY)

# Total in-round attempts (not the same as MAX_ROUNDS). max_retries = retries after attempt 1.
MAX_ATTEMPTS = int(
    os.getenv("MAX_ATTEMPTS", os.getenv("LLM_REFLECTION_MAX_RETRIES", "5"))
)
max_retries = max(0, MAX_ATTEMPTS - 1)
retry_count = 0
inject_failure = os.getenv("INJECT_PERCEPTION_FAILURE", "0") == "1"
inject_affects_motion = os.getenv("INJECT_AFFECTS_MOTION", "0") == "1"
perception_noise_scale = 0.08
if inject_failure:
    if inject_affects_motion:
        logger.info(
            "INJECT_PERCEPTION_FAILURE: XY noise applied to robot motion (demo / stress test)."
        )
    else:
        logger.info(
            "INJECT_PERCEPTION_FAILURE: noise for reflection logging only; motion uses true cube pose."
        )

def is_cube_grasped():
    """Verify if the cube is currently physically grasped by the gripper."""
    try:
        contacts = p.getContactPoints(gripper, cube)
        if len(contacts) < 1:
            return False
        gripper_tcp_pos, _ = get_gripper_tcp_position()
        cube_pos, _ = p.getBasePositionAndOrientation(cube)
        distance = float(np.linalg.norm(np.array(gripper_tcp_pos) - np.array(cube_pos)))
        # Gripper fingers are 2.5cm long, so TCP (centered) to cube center should be small (usually <= 5.5cm)
        if distance > 0.055:
            return False
        return True
    except Exception as e:
        print(f"Error checking if cube is grasped: {e}")
        return False

def pre_rotate_base(target_pos, steps=150):
    """Smoothly rotate the base joint (joint 1) to face the target position while keeping other joints in their current state."""
    try:
        # Find the true base revolute joint explicitly by name.
        # Relying on "first revolute joint encountered" is fragile and can rotate the wrong joint
        # depending on URDF/joint ordering, which then breaks XY alignment convergence.
        base_joint_idx = None
        for j in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, j)
            if info[2] != p.JOINT_REVOLUTE:
                continue
            j_name = info[1].decode("utf-8", errors="ignore").lower()
            link_name = info[12].decode("utf-8", errors="ignore").lower()
            if j_name in ("joint1", "base", "base_joint") or link_name in ("link1",):
                base_joint_idx = j
                break
        if base_joint_idx is None:
            # Fallback: assume joint index 0 is base in many arms
            base_joint_idx = 0

        # Calculate target angle of base joint
        target_angle = math.atan2(target_pos[1], target_pos[0])
        # Clamp to base joint limits (-2.93 to 2.93 rad)
        target_angle = max(-2.93, min(2.93, target_angle))
        
        # Get current joint angles
        start_angle = float(p.getJointState(robot, base_joint_idx)[0])
        
        # Only rotate if the base joint needs to turn significantly (e.g. > 10 degrees)
        angle_diff = abs(start_angle - target_angle)
        if angle_diff < 0.17:  # ~10 degrees
            return

        print(f"[KINEMATICS] Pre-rotating base to face target: {target_angle*180/math.pi:.1f} deg (diff: {angle_diff*180/math.pi:.1f} deg)")

        for step in range(steps):
            alpha = 0.5 * (1.0 - math.cos(math.pi * step / steps))
            interp_angle = (1.0 - alpha) * start_angle + alpha * target_angle
            p.setJointMotorControl2(
                robot,
                base_joint_idx,
                p.POSITION_CONTROL,
                interp_angle,
                force=80.0,
                positionGain=0.1,
                velocityGain=0.4,
            )
            p.stepSimulation()
            time.sleep(1 / 240)
            
        print("[KINEMATICS] Base pre-rotation complete.")
    except Exception as e:
        print(f"Error in pre-rotate base: {e}")

# ---------------- IMPROVED SMOOTH MOTION WITH TCP CALIBRATION ----------------
def smooth_move(
    target_pos,
    steps=500,
    slow_mode=False,
    check_drop=False,
    preserve_orientation=False,
):
    """Improved smooth motion with Cartesian-space interpolation and high-accuracy TCP tracking"""
    joint_indices = []
    joint_types = []
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    
    for j in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, j)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_indices.append(j)
            joint_types.append(joint_info[2])
            lower_limits.append(joint_info[8])
            upper_limits.append(joint_info[9])
            joint_ranges.append(joint_info[9] - joint_info[8])

    rest_poses = [float(p.getJointState(robot, j)[0]) for j in joint_indices]

    # Normalize IK joint targets into feasible ranges.
    # Bullet may clamp out-of-range angles instead of choosing an equivalent wrapped solution,
    # which can freeze a joint (e.g. joint6) and make small TCP moves diverge.
    joint_limits = {
        joint_idx: {
            "type": joint_types[i],
            "lower": float(lower_limits[i]),
            "upper": float(upper_limits[i]),
        }
        for i, joint_idx in enumerate(joint_indices)
    }
    def _adjust_joint_target(joint_idx: int, desired_angle: float) -> float:
        lim = joint_limits[joint_idx]
        lower = lim["lower"]
        upper = lim["upper"]
        jtype = lim["type"]

        # Only do periodic wrapping for revolute joints with ~2pi range.
        if (
            jtype == p.JOINT_REVOLUTE
            and np.isfinite(lower)
            and np.isfinite(upper)
            and (upper - lower) > 5.5
            and (upper - lower) < 7.5
        ):
            period = 2.0 * math.pi
            # Wrap by period into [lower, upper) then clamp.
            wrapped = ((desired_angle - lower) % period) + lower
            return float(np.clip(wrapped, lower, upper))

        # Default: just clamp into joint limits.
        if np.isfinite(lower) and np.isfinite(upper):
            return float(np.clip(desired_angle, lower, upper))
        return float(desired_angle)

    # Get start TCP position
    start_tcp, start_orn = get_gripper_tcp_position()
    target_tcp = np.array(target_pos)

    # Smooth trajectory interpolation in Cartesian space
    for step in range(steps):
        progress = step / steps
        if slow_mode:
            # Cubic easing for very slow, smooth motion
            alpha = progress * progress * (3.0 - 2.0 * progress)
        else:
            # Sine easing for smooth motion
            alpha = 0.5 * (1 - math.cos(math.pi * progress))

        # Linearly interpolate the TCP position in Cartesian space!
        interp_tcp = (1.0 - alpha) * start_tcp + alpha * target_tcp

        move_orn = start_orn if preserve_orientation else GRIPPER_ORIENTATION
        compensated_target, orientation = set_gripper_tcp_target(interp_tcp, orientation=move_orn)

        target_positions = p.calculateInverseKinematics(
            robot,
            ee_index,
            compensated_target.tolist(),
            orientation,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            jointDamping=[0.08] * len(joint_indices),
            maxNumIterations=300,
            residualThreshold=1e-5,
        )

        # Apply smooth joint control with safe force limits and velocity constraints
        force = 80.0  # Stable force for small manipulator
        pos_gain = 0.08 if slow_mode else 0.18
        vel_gain = 0.3 if slow_mode else 0.6
        max_vel = 1.0 if slow_mode else 2.5

        for joint_index in joint_indices:
            # Ensure commanded joint target respects wrap/limit convention.
            desired = float(target_positions[joint_index])
            desired = _adjust_joint_target(joint_index, desired)
            p.setJointMotorControl2(
                robot,
                joint_index,
                p.POSITION_CONTROL,
                desired,
                force=force,
                positionGain=pos_gain,
                velocityGain=vel_gain,
                maxVelocity=max_vel
            )

        # Maintain gripper state — increased holding force to prevent cube dropping during carry.
        for g_joint in gripper_joints:
            p.setJointMotorControl2(
                gripper,
                g_joint,
                p.POSITION_CONTROL,
                targetPosition=target_gripper_joint_pos,
                force=35 if target_gripper_joint_pos == 0.0 else 50,
                positionGain=0.12 if target_gripper_joint_pos == 0.0 else 0.08,
                velocityGain=0.15
            )

        # Check for drop if requested
        if check_drop and not is_cube_grasped():
            print("[SMOOTH MOVE] Aborted: cube dropped!")
            return False

        p.stepSimulation()
        time.sleep(1/240)
        
    # High-fidelity tracking error logging at the end of smooth_move
    print("[DEBUG_TRAJECTORY] Smooth move finished.")
    for j_idx in joint_indices:
        j_state = p.getJointState(robot, j_idx)
        j_name = p.getJointInfo(robot, j_idx)[1].decode('utf-8')
        t_pos = target_positions[j_idx]
        a_pos = j_state[0]
        print(f"  [DEBUG_TRAJECTORY] Joint {j_name}: Target={t_pos:.4f}, Actual={a_pos:.4f}, Error={a_pos - t_pos:.4f}")
    return True

def stabilization_delay(duration=0.5):
    """Add stabilization delay for smooth transitions"""
    print(f"Stabilizing for {duration}s...")
    for _ in range(int(duration * 240)):
        p.stepSimulation()
        time.sleep(1/240)

def align_tcp_over_cube_xy(tolerance_m=0.012, max_nudges=28):
    """Fine XY alignment over the live cube pose before closing the gripper."""
    for nudge_i in range(max_nudges):
        tcp, _ = get_gripper_tcp_position()
        cube_pos, _ = p.getBasePositionAndOrientation(cube)
        err = np.array([cube_pos[0] - tcp[0], cube_pos[1] - tcp[1]])
        dist = float(np.linalg.norm(err))
        if dist <= tolerance_m:
            print(f"TCP–cube XY aligned within {tolerance_m * 1000:.1f} mm (d={dist * 1000:.1f} mm)")
            return True
        # Larger steps when far; cap for stability
        step_cap = 0.020 if dist > 0.04 else (0.014 if dist > 0.020 else 0.008)
        step_xy = np.clip(err, -step_cap, step_cap)
        target_z = max(0.022, float(tcp[2]))
        print(
            f"XY nudge {nudge_i + 1}/{max_nudges}: d={dist * 1000:.1f} mm "
            f"-> dx={step_xy[0]:.4f}, dy={step_xy[1]:.4f}"
        )
        smooth_move(
            [float(tcp[0] + step_xy[0]), float(tcp[1] + step_xy[1]), target_z],
            steps=90,
            slow_mode=True,
            preserve_orientation=True,
        )
        stabilization_delay(0.06)
    tcp, _ = get_gripper_tcp_position()
    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    dist = float(np.linalg.norm(np.array(cube_pos[:2]) - np.array(tcp[:2])))
    print(f"TCP–cube XY alignment finished with residual d={dist * 1000:.1f} mm")
    return dist <= tolerance_m * 1.8


def descend_until_gripper_contact(cxy, z_start, z_floor, max_steps=30):
    """Lower TCP in small steps until gripper and cube report contact, with active floor & top-surface safety stops."""
    z = float(z_start)
    z_floor = float(z_floor)
    for step_i in range(max_steps):
        # 1. Safety check: Stop if gripper makes contact with table/floor to avoid pushing table/cube
        n_floor = len(p.getContactPoints(gripper, plane_id))
        if n_floor > 0:
            print(f"Safety Stop: Table/floor contact detected at z={z:.4f} m! (step {step_i})")
            break

        # 2. Check contact with cube
        cube_contacts = p.getContactPoints(gripper, cube)
        if len(cube_contacts) > 0:
            # Check if any contact normal is pointing vertically (Z-axis), which indicates pushing down on top
            vertical_contact = False
            for contact in cube_contacts:
                normal = contact[7]  # Contact normal vector on child body
                if abs(normal[2]) > 0.7:  # High Z component means vertical contact
                    vertical_contact = True
                    break
            if vertical_contact:
                print(f"Safety Stop: Vertical contact detected at z={z:.4f} m! (Fingers pressing top of cube).")
                break
            else:
                print(f"Lateral Contact during descent: z={z:.4f} m, {len(cube_contacts)} points (step {step_i})")
                return True

        z_next = max(z_floor, z - 0.0018)
        if z_next >= z - 1e-6:
            break
        z = z_next
        smooth_move([float(cxy[0]), float(cxy[1]), z], steps=50, slow_mode=True)
        for _ in range(15):
            p.stepSimulation()
            time.sleep(1 / 240)
    n = len(p.getContactPoints(gripper, cube))
    print(f"Descent finished z={z:.4f} m, contacts={n}")
    return n > 0


# ---------------- STAGED MOVEMENT SEQUENCE ----------------
def check_grasp_contacts_pre_lift():
    """Contacts + TCP–cube alignment while cube may still be on the table."""
    try:
        contacts = p.getContactPoints(gripper, cube)
        print(f"Contact points found: {len(contacts)}")
        if len(contacts) < 1:
            print("No contacts detected")
            return False
        gripper_tcp_pos, _ = get_gripper_tcp_position()
        cube_pos, _ = p.getBasePositionAndOrientation(cube)
        distance = float(np.linalg.norm(np.array(gripper_tcp_pos) - np.array(cube_pos)))
        if distance > 0.10:
            print(f"Cube too far from TCP: {distance:.3f}m")
            return False
        print(f"Pre-lift grasp OK: {len(contacts)} contacts, TCP–cube {distance:.3f}m")
        return True
    except Exception as e:
        print(f"Error in pre-lift grasp check: {e}")
        return False


def mini_lift_grasp_test(cube_xy, pick_policy):
    """Physics check: cube must rise with the arm and stay near TCP (no weld constraint)."""
    cube_pos0, _ = p.getBasePositionAndOrientation(cube)
    z0 = float(cube_pos0[2])
    probe_lift = min(0.05, float(pick_policy["lift_height"]) * 0.22 + 0.02)
    lift_target = [float(cube_xy[0]), float(cube_xy[1]), z0 + probe_lift]
    smooth_move(lift_target, steps=220, slow_mode=True)
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1 / 240)
    cube_pos1, _ = p.getBasePositionAndOrientation(cube)
    z1 = float(cube_pos1[2])
    
    # Enforce robust rise threshold to prevent false positives from vibration/chatter
    min_rise = probe_lift * 0.5
    if z1 - z0 < min_rise:
        print(f"Lift test failed: cube did not rise enough (dz={z1 - z0:.4f}m < threshold={min_rise:.4f}m)")
        return False
    tcp, _ = get_gripper_tcp_position()
    dist = float(np.linalg.norm(np.array(tcp) - np.array(cube_pos1)))
    if dist > 0.10:
        print(f"Lift test failed: cube slipped from TCP (d={dist:.3f}m)")
        return False
    print(f"Lift test OK: dz={z1 - z0:.4f}m, TCP–cube={dist:.3f}m")
    return True


def staged_pick_sequence(
    cube_pos,
    pick_policy,
    local_retry=False,
    *,
    trust_motion_target=True,
):
    """Pick using policy heights. trust_motion_target=False: do not auto re-align to physical cube (eval / LLM+VLA path)."""
    cz = float(cube_pos[2])
    ah = float(pick_policy["approach_height"])
    gh = float(pick_policy["grasp_height"])
    lh = float(pick_policy["lift_height"])
    hover_z = cz + max(ah, gh + 0.02)
    # Ensure all target heights never go below the safe minimum height (0.022) to avoid floor collisions!
    grasp_plane_z = max(0.022, cz + gh)
    fine_z = max(0.022, cz + max(0.002, min(0.012, 0.22 * gh + 0.002)))
    touch_z = max(0.022, cz - 0.002)

    if local_retry:
        print("\n=== LOCAL PICK RETRY (stay near cube — no retreat over goal / high hover) ===")
    else:
        print("\n=== STAGED PICK SEQUENCE START ===")
        pre_rotate_base(cube_pos)

    print("Stage 1: Opening gripper...")
    open_gripper()
    stabilization_delay(0.12 if local_retry else 0.22)

    if not local_retry:
        print(f"Stage 2: Hover z={hover_z:.4f} (approach_height={ah:.4f}, grasp_height={gh:.4f})")
        smooth_move([cube_pos[0], cube_pos[1], hover_z], steps=420, slow_mode=True)
        stabilization_delay(0.35)

        print(f"Stage 3: Descend to grasp plane z={grasp_plane_z:.4f}")
        smooth_move([cube_pos[0], cube_pos[1], grasp_plane_z], steps=520, slow_mode=True)
        stabilization_delay(0.22)

        print(f"Stage 4: Fine approach z={fine_z:.4f}")
        smooth_move([cube_pos[0], cube_pos[1], fine_z], steps=480, slow_mode=True)
        stabilization_delay(0.15)

        print(f"Stage 4.5: Contact z={touch_z:.4f}")
        smooth_move([cube_pos[0], cube_pos[1], touch_z], steps=220, slow_mode=True)
        stabilization_delay(0.15)
    else:
        tcp, _ = get_gripper_tcp_position()
        tz = float(tcp[2])
        print(f"Local retry from TCP z={tz:.4f} (cube z={cz:.4f})")
        # One shortcut lower only if still high; never go up to hover over the goal (red marker) first
        if tz > cz + 0.09:
            bridge_z = max(grasp_plane_z, cz + 0.035)
            print(f"Stage L2a: Shortcut descend to z={bridge_z:.4f} (skip high hover)")
            smooth_move([cube_pos[0], cube_pos[1], bridge_z], steps=260, slow_mode=True)
            stabilization_delay(0.12)
        print(f"Stage L2b: Fine approach z={fine_z:.4f}")
        smooth_move([cube_pos[0], cube_pos[1], fine_z], steps=220, slow_mode=True)
        stabilization_delay(0.1)
        print(f"Stage L2c: Contact z={touch_z:.4f}")
        smooth_move([cube_pos[0], cube_pos[1], touch_z], steps=160, slow_mode=True)
        stabilization_delay(0.1)

    print("Stage 4.6: Ensure gripper–cube contact (descend / nudge if needed)...")
    contacts_before = p.getContactPoints(gripper, cube)
    print(f"Contacts before alignment: {len(contacts_before)}")
    # Prevent plunging too low and hitting the table (finger tips extend 1cm below TCP)
    z_floor = max(0.022, cz - 0.002)
    if len(contacts_before) == 0:
        descend_until_gripper_contact([cube_pos[0], cube_pos[1]], touch_z, z_floor)
    contacts_before = p.getContactPoints(gripper, cube)
    if len(contacts_before) == 0:
        for ox, oy in ((0.006, 0.0), (-0.006, 0.0), (0.0, 0.006), (0.0, -0.006)):
            smooth_move([cube_pos[0] + ox, cube_pos[1] + oy, touch_z], steps=100, slow_mode=True)
            stabilization_delay(0.08)
            descend_until_gripper_contact([cube_pos[0] + ox, cube_pos[1] + oy], touch_z, z_floor)
            if len(p.getContactPoints(gripper, cube)) > 0:
                break
    print(f"Contacts after alignment: {len(p.getContactPoints(gripper, cube))}")

    # Optional: trust the motion target (possibly wrong after bad perception / policy), for pipeline evaluation.
    actual_cube, _ = p.getBasePositionAndOrientation(cube)
    tcp_now, _ = get_gripper_tcp_position()
    xy_gap = float(np.linalg.norm(np.array(actual_cube[:2]) - np.array(tcp_now[:2])))
    if trust_motion_target and xy_gap > 0.015:
        print(
            f"Stage 4.65: Re-center on physical cube (XY gap {xy_gap * 1000:.1f} mm) "
            f"before fine alignment..."
        )
        smooth_move(
            [
                float(actual_cube[0]),
                float(actual_cube[1]),
                max(0.022, float(tcp_now[2])),
            ],
            steps=320,
            slow_mode=True,
            preserve_orientation=True,
        )
        stabilization_delay(0.12)

    print("Stage 4.7: Fine XY alignment over cube...")
    if trust_motion_target:
        align_tcp_over_cube_xy()
        _save_overhead_snapshot("pre_grasp")
    else:
        print(
            "[PIPELINE_EVAL] Skipping auto TCP–cube XY align (policy / VLA must fix mis-aim).",
            flush=True,
        )
        _save_overhead_snapshot("pre_grasp")

    print("Stage 5: Close gripper...")
    # High-fidelity debug state logging
    actual_tcp, actual_orn = get_gripper_tcp_position()
    actual_cube, _ = p.getBasePositionAndOrientation(cube)
    actual_euler = p.getEulerFromQuaternion(actual_orn)
    print(f"[DEBUG] Target grasp TCP: {cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {touch_z:.4f}")
    print(f"[DEBUG] Actual reached TCP: {actual_tcp[0]:.4f}, {actual_tcp[1]:.4f}, {actual_tcp[2]:.4f}")
    print(f"[DEBUG] Actual reached orientation (Euler deg): {[d * 180 / math.pi for d in actual_euler]}")
    print(f"[DEBUG] Actual Cube position: {actual_cube[0]:.4f}, {actual_cube[1]:.4f}, {actual_cube[2]:.4f}")
    print(f"[DEBUG] Distance TCP-Cube: {np.linalg.norm(np.array(actual_tcp) - np.array(actual_cube)):.4f}m")
    
    close_gripper()
    stabilization_delay(0.4)

    print("Stage 6: Pre-lift contact check...")
    if not check_grasp_contacts_pre_lift():
        open_gripper()
        stabilization_delay(0.2)
        return False

    print("Stage 6b: Mini lift grasp test...")
    if not mini_lift_grasp_test(cube_pos, pick_policy):
        open_gripper()
        stabilization_delay(0.2)
        return False

    print(f"Stage 7: Lift to carry height (lift_height={lh:.4f})")
    cube_now, _ = p.getBasePositionAndOrientation(cube)
    lift_target = [float(cube_now[0]), float(cube_now[1]), float(cube_now[2]) + max(lh * 0.5, 0.08)]
    if not smooth_move(lift_target, steps=420, slow_mode=True, check_drop=True):
        print("Staged pick sequence failed: cube dropped during final lift")
        open_gripper()
        stabilization_delay(0.2)
        return False
    stabilization_delay(0.25)
    
    # Final check before concluding pick is successful
    if not is_cube_grasped():
        print("Staged pick sequence failed: cube not grasped at carry pose")
        open_gripper()
        stabilization_delay(0.2)
        return False

    print("=== LOCAL PICK RETRY COMPLETE ===" if local_retry else "=== STAGED PICK SEQUENCE COMPLETE ===")
    return True


def staged_place_sequence(goal_pos, place_policy):
    """Place using same policy height semantics as pick. Returns True if place completed with cube grasped until release."""
    print("\n=== STAGED PLACE SEQUENCE START ===")
    pre_rotate_base(goal_pos)
    gz = float(goal_pos[2])
    ah = float(place_policy["approach_height"])
    gh = float(place_policy["grasp_height"])
    lh = float(place_policy["lift_height"])
    hover_goal_z = gz + max(ah, gh + 0.02)

    print("Stage 1: Moving above goal...")
    if not smooth_move([goal_pos[0], goal_pos[1], hover_goal_z], steps=420, slow_mode=True, check_drop=True):
        print("Aborting place sequence: cube dropped during Stage 1")
        # Retract to safe height at current XY coordinates to clear environment
        curr_tcp, _ = get_gripper_tcp_position()
        smooth_move([curr_tcp[0], curr_tcp[1], gz + lh], steps=200, slow_mode=True)
        return False
    stabilization_delay(0.22)

    print("Stage 2: Descend to release height...")
    if not smooth_move([goal_pos[0], goal_pos[1], gz + gh], steps=520, slow_mode=True, check_drop=True):
        print("Aborting place sequence: cube dropped during Stage 2")
        # Retract to safe height at current XY coordinates to clear environment
        curr_tcp, _ = get_gripper_tcp_position()
        smooth_move([curr_tcp[0], curr_tcp[1], gz + lh], steps=200, slow_mode=True)
        return False
    stabilization_delay(0.22)

    print("Stage 3: Release...")
    open_gripper()
    stabilization_delay(0.35)

    print("Stage 4: Retract...")
    smooth_move([goal_pos[0], goal_pos[1], gz + lh], steps=420, slow_mode=True)
    stabilization_delay(0.22)
    print("=== STAGED PLACE SEQUENCE COMPLETE ===")
    return True


# ---------------- VLA CLOSED-LOOP RECOVERY ----------------
def _vla_notify(message: str, step: int = None, max_steps: int = None):
    """Print, log, and show GUI status while VLA recovery is applying corrections."""
    if step is not None and max_steps is not None:
        line = f"[VLA RECOVERY] Step {step}/{max_steps}: {message}"
        gui_detail = f"Step {step}/{max_steps} — {message}"
    else:
        line = f"[VLA RECOVERY] {message}"
        gui_detail = message
    print(line, flush=True)
    logger.info(line)
    gui_status.update_status("VLA Recovery", gui_detail)


def run_vla_closed_loop_recovery(cube_pos, max_steps=50):
    """
    Executes a real-time, closed-loop recovery sequence guided by the VLARecoveryAgent.
    Runs at 10Hz, calling predict_action at each step, generating delta coordinate shifts,
    solving inverse kinematics, and commanding the robot arm joints.
    """
    print("\n" + "=" * 60, flush=True)
    print("=== VLA CLOSED-LOOP RECOVERY — APPLYING CORRECTIONS ===", flush=True)
    print("=" * 60 + "\n", flush=True)
    _vla_notify("Starting closed-loop recovery (local heuristic VLA)")
    _save_overhead_snapshot("vla_recovery_start")

    _vla_notify("Opening gripper before alignment")
    open_gripper()
    stabilization_delay(0.15)

    # Bridge down toward the cube before pixel-loop nudging (retries often start far above)
    try:
        tcp, _ = get_gripper_tcp_position()
        actual_cube, _ = p.getBasePositionAndOrientation(cube)
        cz = float(actual_cube[2])
        tz = float(tcp[2])
        touch_z = max(0.022, cz - 0.002)
        if tz > cz + 0.04:
            bridge_z = max(touch_z + 0.012, cz + 0.035)
            _vla_notify(f"Bridge descend z={tz:.4f} → {bridge_z:.4f} m")
            smooth_move([float(actual_cube[0]), float(actual_cube[1]), bridge_z], steps=280, slow_mode=True)
            stabilization_delay(0.12)
            fine_z = max(0.022, cz + 0.006)
            smooth_move([float(actual_cube[0]), float(actual_cube[1]), fine_z], steps=180, slow_mode=True)
            stabilization_delay(0.1)
            smooth_move([float(actual_cube[0]), float(actual_cube[1]), touch_z], steps=140, slow_mode=True)
            stabilization_delay(0.1)
            descend_until_gripper_contact([float(actual_cube[0]), float(actual_cube[1])], touch_z, touch_z)
    except Exception as exc:
        print(f"[VLA RECOVERY] Bridge descend skipped: {exc}")
    
    success = False
    best_xy_error = float("inf")
    stall_steps = 0
    
    for step in range(1, max_steps + 1):
        if not p.isConnected():
            print("[VLA RECOVERY] Physics server disconnected; aborting recovery.")
            break
        # A. Capture camera RGB image & get relative error coordinates
        try:
            error_info, rgb = get_relative_pixel_error_overhead_and_rgb(
                target_body_id=cube,
                reference_body_id=gripper,
                verbose=False
            )
        except Exception as exc:
            print(f"[VLA RECOVERY] Camera capture failed at step {step}: {exc}")
            error_info, rgb = None, None
        
        # Fallback to absolute difference if visual tracking fails
        if error_info is None:
            try:
                g_pos, _ = get_gripper_tcp_position()
                c_pos, _ = p.getBasePositionAndOrientation(cube)
                pixel_error_x, pixel_error_y = world_xy_to_overhead_pixel_error(
                    c_pos[0] - g_pos[0], c_pos[1] - g_pos[1]
                )
            except Exception:
                pixel_error_x = 0.0
                pixel_error_y = 0.0
        else:
            pixel_error_x = float(error_info[0])
            pixel_error_y = float(error_info[1])
            
        g_pos, g_orn = get_gripper_tcp_position()
        c_pos, _ = p.getBasePositionAndOrientation(cube)
        
        contacts = p.getContactPoints(gripper, cube)
        contacts_count = len(contacts)
        
        relative_state = {
            "pixel_error_x": pixel_error_x,
            "pixel_error_y": pixel_error_y,
            "gripper_pos": list(g_pos),
            "cube_pos": list(c_pos),
            "contacts_count": contacts_count,
            "recovery_step": step,
            "max_steps": max_steps
        }
        
        xy_err_mm = float(
            np.linalg.norm(np.array(c_pos[:2]) - np.array(g_pos[:2]))
        ) * 1000.0

        # B. Query the VLA agent
        _vla_notify(
            f"Computing correction (TCP–cube XY={xy_err_mm:.1f} mm, contacts={contacts_count})",
            step,
            max_steps,
        )
        action = vla_agent.predict_action(
            rgb=rgb,
            text_instruction="re-align and grasp the cube",
            relative_state=relative_state
        )

        dx = action.get("dx", 0.0)
        dy = action.get("dy", 0.0)
        dz = action.get("dz", 0.0)
        gripper_close = action.get("gripper_close", False)
        terminate = action.get("terminate", False)
        explanation = action.get("explanation", "")

        action_msg = (
            f"Δx={dx * 1000:.1f}mm Δy={dy * 1000:.1f}mm Δz={dz * 1000:.1f}mm "
            f"close={gripper_close} terminate={terminate}"
        )
        _vla_notify(action_msg, step, max_steps)
        if explanation:
            print(f"  [VLA RECOVERY] Reasoning: {explanation}", flush=True)
            logger.info(f"VLA reasoning: {explanation}")

        if terminate:
            _vla_notify("Agent requested stop", step, max_steps)
            break

        # C. Apply translation offsets to TCP target position
        target_x = g_pos[0] + dx
        target_y = g_pos[1] + dy
        target_z = max(0.022, g_pos[2] + dz)

        _vla_notify(
            f"Applying move → TCP ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})",
            step,
            max_steps,
        )
        smooth_move(
            [target_x, target_y, target_z],
            steps=15,
            slow_mode=True,
            preserve_orientation=True,
        )
        
        # D. Step simulation
        for _ in range(5):
            p.stepSimulation()
            time.sleep(1 / 240)

        tcp_after, _ = get_gripper_tcp_position()
        c_after, _ = p.getBasePositionAndOrientation(cube)
        xy_err = float(np.linalg.norm(np.array(c_after[:2]) - np.array(tcp_after[:2])))
        _vla_notify(
            f"Move applied — residual TCP–cube XY={xy_err * 1000:.1f} mm",
            step,
            max_steps,
        )
        if step % 5 == 0:
            _save_overhead_snapshot(f"vla_step_{step}")

        if xy_err < best_xy_error - 0.002:
            best_xy_error = xy_err
            stall_steps = 0
        else:
            stall_steps += 1
        if stall_steps >= 12 and xy_err > 0.018:
            _vla_notify(
                f"Stopping early — XY stalled at {xy_err * 1000:.1f} mm "
                f"(best={best_xy_error * 1000:.1f} mm)",
                step,
                max_steps,
            )
            break
            
        # E. Grasp action handling
        if gripper_close:
            _vla_notify("Closing gripper — grasp attempt", step, max_steps)
            close_gripper()
            stabilization_delay(0.4)
            
            # Run physical verification checks
            if check_grasp_contacts_pre_lift() and mini_lift_grasp_test(c_pos, policy):
                # Grasp successfully completed! Lift to carry altitude
                cube_now, _ = p.getBasePositionAndOrientation(cube)
                lift_z = cube_now[2] + max(policy["lift_height"] * 0.5, 0.08)
                if smooth_move([cube_now[0], cube_now[1], lift_z], steps=150, slow_mode=True, check_drop=True):
                    if is_cube_grasped():
                        _vla_notify("SUCCESS — object secured at carry height", step, max_steps)
                        success = True
                        break

            _vla_notify("Grasp check failed — reopening gripper to re-align", step, max_steps)
            open_gripper()
            stabilization_delay(0.2)
            try:
                c_now, _ = p.getBasePositionAndOrientation(cube)
                touch_z = max(0.022, float(c_now[2]) - 0.002)
                descend_until_gripper_contact([float(c_now[0]), float(c_now[1])], touch_z, touch_z)
            except Exception:
                pass
            
    if not success:
        _vla_notify("FAILED — could not secure grasp within step limit")
        curr_tcp, _ = get_gripper_tcp_position()
        _vla_notify("Retracting to safe height")
        smooth_move([curr_tcp[0], curr_tcp[1], curr_tcp[2] + 0.08], steps=100, slow_mode=True)

    print("\n=== VLA CLOSED-LOOP RECOVERY END ===\n", flush=True)
    logger.info(f"VLA_RECOVERY: finished success={success}")
    return success


# ---------------- ADAPTIVE RETRY LOGIC ----------------
def adaptive_retry_adjustment(failure_type, cube_pos, current_policy):
    """Implement adaptive retry logic with parameter adjustment"""
    print(f"\n=== ADAPTIVE RETRY ANALYSIS ===")
    print(f"Failure type: {failure_type}")
    print(f"Current policy: {current_policy}")
    
    updated_policy = current_policy.copy()
    
    if failure_type == "grasp_failure":
        # Analyze specific grasp failure
        gripper_tcp_pos, _ = get_gripper_tcp_position()
        distance_to_cube = np.linalg.norm(np.array(gripper_tcp_pos) - np.array(cube_pos))
        
        print(f"Distance to cube: {distance_to_cube:.3f}m")
        
        if distance_to_cube > 0.08:
            # Too far from object - adjust approach height
            updated_policy["approach_height"] -= 0.01  # Lower approach
            # Remove blind drift: do not apply blind XY offset guesses
            print("Adjustment: Lower approach height (blind XY drift disabled)")
            
        elif distance_to_cube < 0.03:
            # Too close - adjust grasp height
            updated_policy["grasp_height"] += 0.005  # Raise grasp height
            print("Adjustment: Raise grasp height")
            
        else:
            # Distance OK but grasp failed - adjust gripper force and timing
            updated_policy["release_delay"] += 10  # Increase stabilization
            print("Adjustment: Increase stabilization delay")
            
    elif failure_type == "side_grasp_failure":
        # Side grasp specific adjustments
        updated_policy["grasp_height"] -= 0.005  # Lower for better side contact
        updated_policy["approach_height"] -= 0.01  # Better side approach
        print("Adjustment: Optimize for side grasping")
        
    elif failure_type == "placement_failure":
        # Placement specific adjustments
        updated_policy["lift_height"] += 0.01  # Higher lift for better placement
        updated_policy["release_delay"] += 15  # Longer release delay
        print("Adjustment: Optimize placement sequence")
        
    # Safety bounds checking
    updated_policy["approach_height"] = max(0.05, min(0.20, updated_policy["approach_height"]))
    updated_policy["grasp_height"] = max(-0.015, min(0.05, updated_policy["grasp_height"]))
    updated_policy["lift_height"] = max(0.10, min(0.30, updated_policy["lift_height"]))
    updated_policy["release_delay"] = max(30, min(120, updated_policy["release_delay"]))
    
    print(f"Updated policy: {updated_policy}")
    print("=== END ADAPTIVE RETRY ANALYSIS ===\n")
    
    return updated_policy

# ---------------- GRIPPER FRICTION RECOMMENDATIONS ----------------
def print_gripper_friction_recommendations():
    """Print recommendations for improving gripper finger friction"""
    print("\n=== GRIPPER FRICTION RECOMMENDATIONS ===")
    print("For improved grasping performance, consider:")
    print("1. Rubber pads: Add thin rubber sheets to gripper fingers")
    print("2. Silicone pads: Use silicone grip pads for better friction")
    print("3. Textured surface: Add sandpaper or textured grip tape")
    print("4. 3D printed patterns: Create custom finger textures")
    print("5. Soft foam: Add thin foam layer for conformal grip")
    print("6. Anti-slip tape: Use industrial anti-slip materials")
    print("==========================================\n")

# ---------------- MULTI-ROUND EVALUATION CONFIG ----------------
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "5"))  # Configurable; default 5 rounds per run
current_round = 1
all_rounds_history = []
vla_demo_completed = False

# ---------------- STATE MACHINE ----------------
state = "plan"
timer = 0
stable_counter = 0
last_failure_type = "startup"
last_distance_to_goal = None
attempt_history = []

# Print gripper friction recommendations once
print_gripper_friction_recommendations()

agent = LLMReflectionAgent(
    backend=os.getenv("LLM_AGENT_BACKEND", "ollama"),
    model=os.getenv("LLM_AGENT_MODEL", "llama3.2-vision"),
    endpoint=os.getenv("LLM_AGENT_ENDPOINT", "http://localhost:11434/api/chat"),
    timeout_s=float(os.getenv("LLM_AGENT_TIMEOUT_S", "90.0")),  # Increased to 90s for robust local LLM execution
    use_vision=os.getenv("LLM_AGENT_USE_VISION", "0") == "1"
)
if not USE_LLM_AGENT:
    agent.api_key = None
    agent.endpoint = ""

if USE_LLM_AGENT:
    if agent.is_configured():
        print("LLM agent enabled")
        print("Backend:", agent.backend)
        print("Vision Model:", agent.vision_model)
        print("Reasoning Model:", agent.reasoning_model)
        print("Endpoint:", agent.endpoint)
    else:
        print("LLM agent requested, but configuration is incomplete. Using fallback heuristic.")
else:
    print("LLM agent disabled. Using fallback heuristic.")

# Initialize VLA recovery agent
USE_VLA_RECOVERY = os.getenv("USE_VLA_RECOVERY", "0") == "1"
vla_agent = None
if USE_VLA_RECOVERY:
    vla_agent = VLARecoveryAgent(
        backend=os.getenv("VLA_BACKEND", "heuristic"),
        api_url=os.getenv("VLA_API_URL", "http://localhost:8000/predict"),
        model_path=os.getenv("VLA_MODEL_PATH") or None,
    )
    print(f"VLA recovery enabled (backend={vla_agent.backend})")
else:
    print("VLA Hybrid Recovery Agent disabled.")

if FORCE_REFLECTION:
    print("Force reflection enabled for", FORCED_REFLECTION_ATTEMPTS, "attempt(s)")
if REFLECT_EVERY_ROUND:
    print("LLM reflection after each completed round: enabled")
else:
    print("LLM reflection: on failure only (in-round retries)")
print(
    f"Task: {MAX_ROUNDS} round(s), {MAX_ATTEMPTS} attempt(s) per round "
    f"(LLM + VLA on failure only)"
)
if USE_VLA_RECOVERY and VLA_ON_ANY_PICK_FAILURE:
    print("VLA recovery on any staged-pick failure: enabled")
if VLA_DEMO_MODE:
    print(
        "VLA demo mode: after first successful staged pick in a session, "
        "gripper opens and closed-loop VLA runs (set VLA_DEMO_MODE=0 for normal eval)"
    )


def run_robot_reflection_analysis(
    reflection_type,
    *,
    snapshot_tag=None,
    apply_updates=True,
    record_history=True,
):
    """Run LLM reflection (Ollama or heuristic fallback) and optionally update policy."""
    global policy, attempt_history

    tag = snapshot_tag or f"reflect_a{retry_count + 1}"
    attempt_num = retry_count + 1
    max_attempts = MAX_ATTEMPTS
    failure_labels = {
        "grasp_failure": "grasp failed (pick or carry lost)",
        "cube_out_of_reach": "cube left reachable workspace",
        "placement_failure": "place failed",
        "observe_timeout": "object did not stabilize at goal in time",
        "forced_reflection": "forced reflection (debug)",
    }
    failure_label = failure_labels.get(reflection_type, reflection_type)

    print("\n" + "=" * 62)
    print(f"LLM REFLECTION — after attempt {attempt_num} of {max_attempts}")
    print(f"Failure: {failure_label}")
    print("=" * 62)
    gui_status.update_status(
        "Reflecting",
        f"Attempt {attempt_num}/{max_attempts}: {failure_label[:50]}",
    )
    gui_status.display_status()
    log_robot_state(
        logger,
        "REFLECTING",
        f"attempt {attempt_num}/{max_attempts} failure={reflection_type}",
        attempt_num,
    )

    error, rgb = None, None
    if p.isConnected():
        try:
            error, rgb = get_relative_pixel_error_overhead_and_rgb(
                target_body_id=cube,
                reference_body_id=gripper,
                verbose=False,
            )
        except Exception as exc:
            print(f"Overhead capture for reflection failed: {exc}")
    else:
        print("Physics server disconnected — skipping overhead capture for reflection.")

    _save_overhead_snapshot(tag)

    cube_visible = error is not None
    offline_summary = None
    offline_confidence = None

    if error is None:
        print("Overhead view occluded -> computing physical coordinate-based pixel error")
        try:
            gripper_pos, _ = get_gripper_tcp_position()
            cube_pos, _ = p.getBasePositionAndOrientation(cube)
            pixel_error_x, pixel_error_y = world_xy_to_overhead_pixel_error(
                cube_pos[0] - gripper_pos[0],
                cube_pos[1] - gripper_pos[1],
            )
        except Exception:
            pixel_error_x = 0.0
            pixel_error_y = 0.0
    else:
        pixel_error_x, pixel_error_y = float(error[0]), float(error[1])

    if offline_classifier is not None and rgb is not None:
        try:
            pred = offline_classifier.predict(rgb)
            offline_summary = pred.label
            offline_confidence = float(pred.confidence)
            print(f"OfflineVLM: {pred.label} (conf={pred.confidence:.2f})")
        except Exception as exc:
            print("OfflineVLM prediction failed:", exc)

    joint_positions = []
    if p.isConnected():
        for j in range(p.getNumJoints(robot)):
            joint_info = p.getJointInfo(robot, j)
            if joint_info[2] == p.JOINT_REVOLUTE:
                joint_positions.append(float(p.getJointState(robot, j)[0]))

    scene_info = {
        "failure_type": reflection_type,
        "attempt_number": attempt_num,
        "max_attempts": max_attempts,
        "retry_count": int(retry_count),
        "cube_visible": bool(cube_visible),
        "pixel_error_x": float(pixel_error_x),
        "pixel_error_y": float(pixel_error_y),
        "distance_to_goal": None if last_distance_to_goal is None else float(last_distance_to_goal),
        "offline_direction_label": offline_summary,
        "offline_direction_confidence": offline_confidence,
        "joint_positions": joint_positions,
    }

    decision = agent.reflect(
        scene_info=scene_info,
        policy=policy,
        rgb=rgb,
        history=attempt_history,
    )

    print("-" * 62)
    print(f"LLM backend: {decision.mode}")
    if decision.confidence is not None:
        print(f"Confidence: {decision.confidence:.2f}")
    print(f"Pixel error (overhead): x={pixel_error_x:.1f} px, y={pixel_error_y:.1f} px")
    print("-" * 62)
    print("LLM explanation:")
    print(decision.explanation)
    print("-" * 62)
    print("Policy deltas:", decision.updates if decision.updates else "(none)")
    if decision.terminate:
        print("LLM decision: STOP (terminate=true)")
    else:
        next_attempt = min(attempt_num + 1, max_attempts)
        print(f"LLM decision: RETRY → starting attempt {next_attempt} of {max_attempts}")
    print("=" * 62 + "\n")

    log_llm_decision(logger, decision)

    confidence_str = f"{decision.confidence:.2f}" if decision.confidence is not None else "N/A"
    explanation_str = decision.explanation[:100] + (
        "..." if len(decision.explanation) > 100 else ""
    )
    llm_summary = f"Mode: {decision.mode} | Confidence: {confidence_str}\nExplanation: {explanation_str}"
    gui_status.update_status("Reflecting", "LLM analysis complete", llm_summary)
    gui_status.display_status()

    if apply_updates and decision.updates:
        old_policy = policy.copy()
        policy = apply_policy_updates(policy, decision.updates)
        log_policy_update(logger, old_policy, policy)
        print("Updated policy:", policy)

    if record_history:
        attempt_history.append(
            {
                "retry": int(retry_count),
                "failure_type": reflection_type,
                "cube_visible": bool(cube_visible),
                "pixel_error_x": float(pixel_error_x),
                "pixel_error_y": float(pixel_error_y),
                "distance_to_goal": None
                if last_distance_to_goal is None
                else float(last_distance_to_goal),
                "updates": decision.updates,
                "mode": decision.mode,
            }
        )

    return decision


def reflect_after_round_completion(*, place_ok, grasp_ok):
    """LLM reflection between autonomous rounds (does not consume in-round retry budget)."""
    if not REFLECT_EVERY_ROUND:
        return None
    reflection_type = "round_success" if place_ok else "round_failure"
    if grasp_ok and not place_ok:
        reflection_type = "round_grasp_ok_place_fail"
    print(
        f"\n=== ROUND {current_round} REFLECTION "
        f"(place={'OK' if place_ok else 'FAIL'}, grasp={'OK' if grasp_ok else 'FAIL'}) ==="
    )
    return run_robot_reflection_analysis(
        reflection_type,
        snapshot_tag=f"reflect_round_{current_round}",
        apply_updates=False,
        record_history=False,
    )


# ---------------- LOGGING ----------------
attempt_distances = []
successful_grasp = False
successful_placement = False

while p.isConnected():
    timer += 1
    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    cube_pos = np.array(cube_pos)

    # Abort if physics/slippage moved the cube outside the reachable annulus.
    # Skip during "observe" and "done" — in observe the cube may be resting at the
    # goal which can be near the base; that is intentional, not an error.
    if state not in ("observe", "done"):
        reach_ok, reach_dist, reach_msg = check_cube_in_workspace(cube_pos, label="cube")
        if not reach_ok:
            report_cube_out_of_reach(reach_msg, cube_pos, reach_dist)
            # Do not continue here when entering analyze/done — otherwise analyze never runs.
            if state not in ("analyze", "done"):
                continue
    
    # Calculate current distance to goal
    goal_pos, _ = p.getBasePositionAndOrientation(goal)
    current_distance_to_goal = float(np.linalg.norm(np.array(cube_pos) - np.array(goal_pos)))
    
    # Update robot joint angles in GUI only (no screen display, no parameters)
    update_robot_joint_angles()
    
    # Update GUI metrics
    gui_status.update_metrics(retry_count + 1, current_distance_to_goal)
        
    if state == "plan":
        successful_grasp = False
        successful_placement = False
        print("\nPlanning attempt", retry_count + 1)
        
        # Check if goal position is reachable
        goal_distance_from_robot = np.linalg.norm(goal_position[:2] - np.array([0, 0]))
        if goal_distance_from_robot > REACHABLE_THRESHOLD:
            print(f"WARNING: Goal position too far from robot!")
            print(f"Goal distance: {goal_distance_from_robot:.3f}m, Max reach: {REACHABLE_THRESHOLD:.3f}m")
            
            # Log unreachable goal
            logger.info(f"UNREACHABLE GOAL: Distance {goal_distance_from_robot:.3f}m exceeds reach {REACHABLE_THRESHOLD:.3f}m")
            
            # Update GUI with unreachable goal status
            gui_status.update_status("Unreachable", f"Goal too far: {goal_distance_from_robot:.3f}m > {REACHABLE_THRESHOLD:.3f}m")
            gui_status.display_status()
            
            # Terminate execution
            state = "done"
            continue
        
        gui_status.update_status("Planning", f"Attempt {retry_count + 1}")
        gui_status.update_metrics(retry_count + 1, last_distance_to_goal)
        gui_status.display_status()
        
        # Log planning state
        log_robot_state(logger, "PLANNING", f"Attempt {retry_count + 1}", retry_count + 1, last_distance_to_goal)

        # Query active physical cube coordinates to dynamically compensate for shifting/slippage
        try:
            actual_pos, _ = p.getBasePositionAndOrientation(cube)
            perceived_cube_pos = np.array(actual_pos)
        except Exception:
            perceived_cube_pos = cube_pos.copy()

        if PIPELINE_EVAL_MODE and retry_count == 0:
            mx = float(os.getenv("PIPELINE_EVAL_MISS_X_M", "0.022"))
            my = float(os.getenv("PIPELINE_EVAL_MISS_Y_M", "-0.018"))
            perceived_cube_pos[0] += mx
            perceived_cube_pos[1] += my
            print(
                f"[PIPELINE_EVAL] Attempt 1 deliberate aim bias Δxy=({mx:.4f}, {my:.4f}) m "
                f"(offsets still apply)"
            )
            logger.info(f"PIPELINE_EVAL_MISS xy=({mx:.4f},{my:.4f})")

        reach_ok, reach_dist, reach_msg = check_cube_in_workspace(perceived_cube_pos, label="cube")
        if not reach_ok:
            report_cube_out_of_reach(reach_msg, perceived_cube_pos, reach_dist)
            if state not in ("analyze", "done"):
                continue

        perceived_cube_pos[0] += policy["x_offset"]
        perceived_cube_pos[1] += policy["y_offset"]

        if inject_failure:
            offset_x = 0.025
            offset_y = -0.020
            if inject_affects_motion:
                print("[PERCEPTION] Injecting misalignment into motion target (attempt 1)!")
                perceived_cube_pos[0] += offset_x
                perceived_cube_pos[1] += offset_y
                print(f"[PERCEPTION] Motion target shifted by X={offset_x:.3f}m, Y={offset_y:.3f}m")
            else:
                print(
                    "[PERCEPTION] Simulated perception error for reflection only "
                    f"(motion uses true cube; inject XY=({offset_x:.3f}, {offset_y:.3f}))"
                )

        # Enforce base joint revolute limit safety clamp on perceived cube coordinates to avoid dead-zone near 180 degrees
        perceived_angle = math.atan2(perceived_cube_pos[1], perceived_cube_pos[0])
        if abs(perceived_angle) > 2.79:
            print(f"[KINEMATICS] Clamping perceived cube angle from {perceived_angle*180/math.pi:.1f} deg to {math.copysign(2.79, perceived_angle)*180/math.pi:.1f} deg to respect mechanical limits")
            perceived_angle = math.copysign(2.79, perceived_angle)
            perceived_dist = math.sqrt(perceived_cube_pos[0]**2 + perceived_cube_pos[1]**2)
            perceived_cube_pos[0] = perceived_dist * math.cos(perceived_angle)
            perceived_cube_pos[1] = perceived_dist * math.sin(perceived_angle)

        approach_target = perceived_cube_pos.copy()
        approach_target[2] += policy["approach_height"]

        grasp_target = perceived_cube_pos.copy()
        grasp_target[2] += policy["grasp_height"]
        logger.info(
            f"Plan targets: approach_z={approach_target[2]:.4f} grasp_z={grasp_target[2]:.4f} "
            f"offsets xy=({policy['x_offset']:.4f},{policy['y_offset']:.4f})"
        )

        state = "approach"
        timer = 0

    elif state == "approach":
        _save_overhead_snapshot(f"approach_r{retry_count + 1}")
        gui_status.update_status("Approaching", "Using staged pick sequence")
        gui_status.display_status()
        log_robot_state(logger, "APPROACHING", "Using staged pick sequence", retry_count + 1)
        
        # Use new staged pick sequence or VLA closed-loop recovery
        local_pick_retry = last_failure_type == "grasp_failure" and retry_count > 0
        if local_pick_retry:
            logger.info("LOCAL_PICK_RETRY: continuing from near-cube pose (no full re-hover)")
        
        trust_motion_target = (
            not PIPELINE_EVAL_MODE
            or retry_count >= max_retries
            or local_pick_retry
        )
        if PIPELINE_EVAL_MODE and not trust_motion_target:
            print(
                f"[PIPELINE_EVAL] Attempt {retry_count + 1}/{MAX_ATTEMPTS}: "
                "keeping mis-aim for staged pick (no physical auto-realign).",
                flush=True,
            )
        grasp_success = staged_pick_sequence(
            perceived_cube_pos,
            policy,
            local_retry=local_pick_retry,
            trust_motion_target=trust_motion_target,
        )

        if (
            VLA_DEMO_MODE
            and grasp_success
            and USE_VLA_RECOVERY
            and vla_agent is not None
            and not vla_demo_completed
            and retry_count == 0
            and not PIPELINE_EVAL_MODE
        ):
            print(
                "\n[VLA DEMO] Staged pick OK — opening gripper to exercise VLA recovery "
                "(arm stays near cube; not a real failure)\n",
                flush=True,
            )
            logger.info("VLA_DEMO: triggering closed-loop recovery after successful staged pick")
            open_gripper()
            stabilization_delay(0.25)
            grasp_success = False
            vla_demo_completed = True

        try_vla_recovery = (
            USE_VLA_RECOVERY
            and vla_agent is not None
            and not grasp_success
            and (VLA_ON_ANY_PICK_FAILURE or local_pick_retry)
        )
        if try_vla_recovery:
            print("\n[VLA] Staged pick failed — engaging closed-loop recovery\n", flush=True)
            logger.info("VLA_RECOVERY: staged pick failed, entering closed-loop recovery")
            try:
                actual_cube, _ = p.getBasePositionAndOrientation(cube)
                vla_target = np.array(actual_cube)
            except Exception:
                vla_target = perceived_cube_pos
            grasp_success = run_vla_closed_loop_recovery(vla_target)
        
        if grasp_success:
            successful_grasp = True
            state = "lift"
            timer = 0
            print("Staged pick sequence completed successfully!")
        else:
            print("Staged pick sequence failed!")
            successful_grasp = False
            last_failure_type = "grasp_failure"
            try:
                c_pos, _ = p.getBasePositionAndOrientation(cube)
                g_pos, _ = p.getBasePositionAndOrientation(goal)
                dist = float(np.linalg.norm(np.array(c_pos) - np.array(g_pos)))
            except Exception:
                dist = last_distance_to_goal or 0.0
            attempt_distances.append(dist)
            state = "analyze"
            timer = 0

    elif state == "lift":
        gui_status.update_status("Lifting", "Moving to place position")
        gui_status.display_status()
        log_robot_state(logger, "LIFTING", "Moving to place position", retry_count + 1)
        
        # Use new staged place sequence with proper TCP calibration
        place_success = staged_place_sequence(goal_position, policy)

        if place_success:
            state = "observe"
            timer = 0
            stable_counter = 0
            print("Staged place sequence completed successfully!")
        else:
            print("Place sequence aborted due to carriage drop!")
            successful_grasp = False
            last_failure_type = "grasp_failure"
            try:
                c_pos, _ = p.getBasePositionAndOrientation(cube)
                g_pos, _ = p.getBasePositionAndOrientation(goal)
                dist = float(np.linalg.norm(np.array(c_pos) - np.array(g_pos)))
            except Exception:
                dist = last_distance_to_goal or 0.0
            attempt_distances.append(dist)
            state = "analyze"
            timer = 0

    elif state == "observe":
        observe_timer = 0
        observe_timeout_frames = int(os.getenv("OBSERVE_TIMEOUT_FRAMES", "3600"))  # 15 s @ 240 Hz
        while p.isConnected() and observe_timer < observe_timeout_frames:
            try:
                # Get current cube position with error handling
                cube_pos, cube_orn = p.getBasePositionAndOrientation(cube)
                goal_pos, _ = p.getBasePositionAndOrientation(goal)
            except Exception as e:
                print(f"Error getting object positions: {e}")
                # Use last known positions or defaults
                cube_pos = [0.2, 0.0, 0.02]  # Default center
                goal_pos = [0.0, 0.0, 0.025]  # Default goal
                print("Using default positions for observation")
            
            # Calculate distance to goal
            distance_to_goal = float(np.linalg.norm(np.array(cube_pos) - np.array(goal_pos)))
            last_distance_to_goal = distance_to_goal

            # Calculate cube height and surface check
            cube_z = cube_pos[2]
            surface_z = 0.025  # Table surface height
            is_on_surface = abs(cube_z - surface_z) < 0.01
            
            try:
                linear_velocity, _ = p.getBaseVelocity(cube)
                speed = float(np.linalg.norm(linear_velocity))
            except Exception:
                speed = 0.0

            # Success criteria: close to goal, low speed, AND on surface
            if distance_to_goal < 0.10 and speed < 0.05 and is_on_surface:
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter > 30:  # Stable for 30 frames
                if FORCE_REFLECTION and retry_count < FORCED_REFLECTION_ATTEMPTS:
                    print("FORCE_REFLECTION active -> sending successful attempt to reflection")
                    attempt_distances.append(distance_to_goal)
                    last_failure_type = "forced_reflection"
                    state = "analyze"
                else:
                    print("SUCCESS -> task completed (object properly placed on surface)")
                    successful_placement = True
                    state = "done"
                    attempt_distances.append(distance_to_goal)
                break  # Exit observe loop

            observe_timer += 1
            timer += 1
            p.stepSimulation()
            time.sleep(1/240)  # Maintain simulation step rate

        # Timeout fallback
        if observe_timer >= observe_timeout_frames:
            print(
                f"Observe timeout reached after {observe_timer} frames "
                f"({observe_timer / 240.0:.1f}s)"
            )
            attempt_distances.append(distance_to_goal)
            last_failure_type = "observe_timeout"
            state = "analyze"

    elif state == "analyze":
        if retry_count >= max_retries:
            print(
                f"\n*** All {MAX_ATTEMPTS} attempt(s) used — ending task "
                f"(last failure: {last_failure_type}) ***\n",
                flush=True,
            )
            state = "done"
            continue

        if last_failure_type == "cube_out_of_reach":
            try:
                reseat_cube_in_workspace()
                cube_reach_error_reported = False
                print("[RECOVERY] Cube re-seated in workspace before reflection retry.")
            except Exception as exc:
                print(f"[RECOVERY] Could not re-seat cube: {exc}")
                state = "done"
                continue

        decision = run_robot_reflection_analysis(
            last_failure_type,
            snapshot_tag=f"reflect_r{retry_count + 1}",
            apply_updates=True,
            record_history=True,
        )

        if decision.terminate:
            print("Agent requested termination")
            state = "done"
            continue

        inject_failure = False
        retry_count += 1
        cube_reach_error_reported = False
        print(
            f"\n>>> Retrying — attempt {retry_count + 1} of {MAX_ATTEMPTS} "
            f"(policy updated by LLM)\n",
            flush=True,
        )
        state = "plan"
        timer = 0

    elif state == "done":
        actual_attempts = retry_count + 1
        success = successful_placement
        
        # Record the outcome of the current round
        round_record = {
            "round": current_round,
            "success": success,
            "grasp_success": successful_grasp,
            "attempts": actual_attempts,
            "final_distance": last_distance_to_goal or 0.0,
            "attempt_distances": list(attempt_distances),
        }
        all_rounds_history.append(round_record)
        
        # Log the round outcome in session logs
        logger.info(
            f"TASK_OUTCOME place_ok={success} grasp_ok={successful_grasp} "
            f"attempts={actual_attempts} d_goal_m={last_distance_to_goal}"
        )
        log_session_summary(
            logger,
            actual_attempts,
            last_distance_to_goal or 0.0,
            success,
            grasp_success=successful_grasp,
            gripper_model=GRIPPER_MODEL_NAME,
            failure_type="" if success else last_failure_type,
        )

        # Log to local SQLite database
        try:
            from utils.logger import log_failure
            joint_positions = []
            for j in range(p.getNumJoints(robot)):
                joint_info = p.getJointInfo(robot, j)
                if joint_info[2] == p.JOINT_REVOLUTE:
                    joint_positions.append(float(p.getJointState(robot, j)[0]))

            try:
                gripper_pos, _ = get_gripper_tcp_position()
                c_pos, _ = p.getBasePositionAndOrientation(cube)
                pixel_error_x, pixel_error_y = world_xy_to_overhead_pixel_error(
                    c_pos[0] - gripper_pos[0],
                    c_pos[1] - gripper_pos[1],
                )
            except Exception:
                pixel_error_x = 0.0
                pixel_error_y = 0.0

            scene_info = {
                "failure_type": "placed_successfully" if success else last_failure_type,
                "retry_count": int(retry_count),
                "cube_visible": True,
                "pixel_error_x": pixel_error_x,
                "pixel_error_y": pixel_error_y,
                "distance_to_goal": None if last_distance_to_goal is None else float(last_distance_to_goal),
                "joint_positions": joint_positions,
            }
            robot_state = {"scene_info": scene_info, "current_policy": policy}
            llm_response = {
                "explanation": "Task completed successfully: object placed on target surface." if success else f"Max retries reached. Last failure: {last_failure_type}",
                "updates": {},
                "terminate": True,
            }
            log_failure(
                failure_type="placed_successfully" if success else last_failure_type,
                robot_state=robot_state,
                llm_response=llm_response,
                strategy_chosen="session_complete",
            )
        except Exception as dbe:
            print(f"[DATABASE ERROR] Could not save round state: {dbe}")

        print(
            f"RESULT ROUND {current_round}: place={'OK' if success else 'FAIL'} grasp={'OK' if successful_grasp else 'FAIL'} "
            f"attempts={actual_attempts} dist_goal={(last_distance_to_goal or 0.0):.3f}m"
        )

        reflect_after_round_completion(place_ok=success, grasp_ok=successful_grasp)
        
        # Decide whether to autonomously trigger the next round
        if current_round < MAX_ROUNDS:
            print(f"\n=============================================================")
            print(f"=== ROUND {current_round} COMPLETED (success={success}, dist={(last_distance_to_goal or 0.0):.3f}m) ===")
            print(f"=== AUTONOMOUSLY INITIALIZING ROUND {current_round + 1} ===")
            print(f"=============================================================\n")
            
            current_round += 1
            
            # Re-generate randomized cube and goal positions for the new round
            optimal_x, optimal_y = calculate_optimal_cube_position()
            goal_x, goal_y = calculate_optimal_goal_position()
            goal_position = np.array([goal_x, goal_y, 0.02])
            
            # Reset visual/physical assets to the new random coordinates
            p.resetBasePositionAndOrientation(cube, [optimal_x, optimal_y, 0.02], [0, 0, 0, 1])
            p.resetBaseVelocity(cube, [0, 0, 0], [0, 0, 0])
            ok_new, dist_new, msg_new = check_cube_in_workspace([optimal_x, optimal_y])
            if not ok_new:
                raise CubeOutOfReachError(msg_new)
            p.resetBasePositionAndOrientation(goal, [goal_x, goal_y, 0.02], [0, 0, 0, 1])
            
            # Reset robot to home posture & open the gripper
            preset_robot_joints_to_home()
            open_gripper()
            
            # Reset variables for the next round
            retry_count = 0
            attempt_distances = []
            attempt_history = []
            successful_grasp = False
            successful_placement = False
            last_failure_type = "startup"
            last_distance_to_goal = None
            cube_reach_error_reported = False
            inject_failure = os.getenv("INJECT_PERCEPTION_FAILURE", "0") == "1"
            
            # Reset offsets back to baseline policy for the new test round
            policy = dict(DEFAULT_POLICY)
            
            # Restart planning autonomously
            state = "plan"
            timer = 0
            continue
        else:
            print(f"\n=============================================================")
            print(f"=== ALL {MAX_ROUNDS} AUTONOMOUS ROUNDS COMPLETED ===")
            print(f"=============================================================")
            for r in all_rounds_history:
                print(f"Round {r['round']}: success={r['success']}, attempts={r['attempts']}, final_dist={r['final_distance']:.3f}m")
            print(f"=============================================================\n")
            break

    p.stepSimulation()
    time.sleep(1/240)  # Maintain simulation step rate

p.disconnect()

# Close GUI status window
gui_status.close()

# ---------------- PLOT RESULTS ----------------
if all_rounds_history:
    plt.figure(figsize=(10, 6))
    
    # Plot learning curve for each round
    for r in all_rounds_history:
        r_dists = r.get("attempt_distances", [r["final_distance"]])
        r_attempts = np.arange(1, len(r_dists) + 1)
        plt.plot(r_attempts, r_dists, marker="o", label=f"Round {r['round']} (Success: {r['success']})")
        
    plt.xlabel("Attempt")
    plt.ylabel("Final distance to goal (m)")
    plt.title("Distance to goal vs. attempt with LLM reflection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_dir = os.path.join(BASE_DIR, "data", "plots")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(out_dir, f"multi_round_evaluation_{timestamp}.png")
    csv_path = os.path.join(out_dir, f"multi_round_evaluation_{timestamp}.csv")

    plt.savefig(png_path, dpi=200)
    plt.close()

    with open(csv_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("round,attempt,final_distance_m,success\n")
        for r in all_rounds_history:
            r_dists = r.get("attempt_distances", [r["final_distance"]])
            for attempt_idx, distance in enumerate(r_dists, 1):
                file_obj.write(f"{r['round']},{attempt_idx},{float(distance)},{r['success']}\n")

    print("Saved combined evaluation plot:", png_path)
    print("Saved combined evaluation data:", csv_path)
