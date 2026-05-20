#!/usr/bin/env python3
"""
Autonomous Robot Reflection Experiment with Intelligent Object Detection and Placement
"""
import os
import sys
import time

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pybullet as p  # type: ignore
import pybullet_data  # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.segmentation import get_relative_pixel_error_overhead_and_rgb  # type: ignore
from perception.autonomous_navigation import AutonomousNavigator  # type: ignore
from reflection.llm_reflection_agent import LLMReflectionAgent, apply_policy_updates  # type: ignore

# ---------------- CONFIGURATION ----------------
USE_LLM_AGENT = os.getenv("USE_LLM_AGENT", "1") == "1"
FORCE_REFLECTION = os.getenv("FORCE_REFLECTION", "1") == "1"
FORCED_REFLECTION_ATTEMPTS = int(os.getenv("FORCED_REFLECTION_ATTEMPTS", "3"))

# Set Ollama as default backend with Llama3.2 Vision model
os.environ["LLM_AGENT_BACKEND"] = "ollama"
os.environ["LLM_AGENT_MODEL"] = "llama3.2-vision"
os.environ["LLM_AGENT_ENDPOINT"] = "http://localhost:11434/api/chat"
os.environ["LLM_AGENT_TIMEOUT_S"] = "60"
os.environ["LLM_AGENT_USE_VISION"] = "1"

# Autonomous navigation settings
TASK_CONTEXT = os.getenv("TASK_CONTEXT", "organize")  # "organize", "sort", "clear"
MAX_OBJECTS = 5
OBJECT_TYPES = ["cube_small", "box", "sphere"]

# ---------------- SIMULATION SETUP ----------------
os.environ["ROBOT_SESSION_ID"] = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

p.setPhysicsEngineParameter(numSolverIterations=150)
p.setPhysicsEngineParameter(fixedTimeStep=1 / 240)

# ---------------- ENVIRONMENT ----------------
plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=1.5)

# ---------------- LOAD ROBOT ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

robot = p.loadURDF(
    URDF_PATH,
    [0.35, 0.0, 0.02],
    [0, 0, 0, 1],
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE,
)

# ---------------- ROBOT CONFIGURATION ----------------
ee_index = 8
joints = []

for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    joint_type = info[2]
    if joint_type != p.JOINT_FIXED:
        joints.append(i)

# Set initial joint positions
home_joints = [0, -np.pi/4, np.pi/4, -np.pi/4, np.pi/4, 0]
for i, joint_id in enumerate(joints):
    p.setJointMotorControl2(robot, joint_id, p.POSITION_CONTROL, home_joints[i])

# ---------------- GRIPPER ----------------
gripper = p.loadURDF(
    os.path.join(BASE_DIR, "urdf", "simple_gripper.urdf"),
    [0, 0, 0],
    [0, 0, 0, 1],
    useFixedBase=False,
)

p.createConstraint(
    robot,
    ee_index,
    gripper,
    -1,
    p.JOINT_FIXED,
    [0, 0, 0],
    [0, 0, 0.04],
    [0, 0, 0],
)

# ---------------- AUTONOMOUS NAVIGATOR ----------------
navigator = AutonomousNavigator()

# Global cube tracker for compatibility
cube = None

# ---------------- SPAWN RANDOM OBJECTS ----------------
def spawn_random_objects(num_objects: int):
    """Spawn random objects in the workspace"""
    spawned_objects = []
    
    for i in range(num_objects):
        # Random position within workspace
        x = np.random.uniform(0.15, 0.45)
        y = np.random.uniform(-0.2, 0.2)
        z = 0.02
        
        # Random object type - use pybullet_data objects
        obj_type = np.random.choice(OBJECT_TYPES)
        
        if obj_type == "cube_small":
            obj = p.loadURDF("cube.urdf", [x, y, z])
            p.changeVisualShape(obj, -1, rgbaColor=[0.2, 0.2, 0.2, 1])
        elif obj_type == "box":
            obj = p.loadURDF("cube.urdf", [x, y, z])
            p.changeVisualShape(obj, -1, rgbaColor=[0.8, 0.4, 0.2, 1])
        else:  # sphere
            obj = p.loadURDF("sphere_small.urdf", [x, y, z])
            p.changeVisualShape(obj, -1, rgbaColor=[0.2, 0.4, 0.8, 1])
        
        p.changeDynamics(obj, -1, lateralFriction=1.5, linearDamping=0.4, angularDamping=0.4)
        spawned_objects.append(obj)
    
    return spawned_objects

# ---------------- HELPER FUNCTIONS ----------------
def smooth_move(target_pos, speed=0.02):
    """Smooth movement to target position"""
    current_pos = p.getLinkState(robot, ee_index)[0]
    distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
    
    steps = int(distance / speed)
    for step in range(steps):
        t = (step + 1) / steps
        interp_pos = current_pos * (1 - t) + np.array(target_pos) * t
        
        # Calculate joint positions using inverse kinematics (simplified)
        joint_positions = p.calculateInverseKinematics(
            robot, ee_index, interp_pos, maxNumIterations=100
        )
        
        for i, joint_id in enumerate(joints):
            p.setJointMotorControl2(robot, joint_id, p.POSITION_CONTROL, joint_positions[i])
        
        p.stepSimulation()
        time.sleep(1/240)

def detect_and_select_target():
    """Detect objects and select target using AI"""
    print("\n🔍 Detecting objects in workspace...")
    
    # Exclude robot and gripper from detection
    exclude_ids = [robot, gripper]
    detected_objects = navigator.detect_all_objects(exclude_body_ids=exclude_ids)
    
    print(f"Found {len(detected_objects)} objects:")
    for i, obj in enumerate(detected_objects):
        print(f"  {i+1}. {obj['type']} at {obj['position']}")
    
    # Select best target object
    target_object = navigator.select_target_object(detected_objects)
    
    if target_object:
        print(f"🎯 Selected target: {target_object['type']} at {target_object['position']}")
        return target_object, detected_objects
    else:
        print("❌ No suitable objects found!")
        return None, detected_objects

def plan_intelligent_placement(target_object, detected_objects):
    """Plan intelligent placement location"""
    print(f"🧠 Planning placement for {target_object['type']}...")
    
    goal_position = navigator.find_optimal_placement_location(
        target_object, detected_objects, TASK_CONTEXT
    )
    
    print(f"📍 Planned placement: {goal_position}")
    return goal_position

# ---------------- LLM AGENT ----------------
llm_agent = LLMReflectionAgent(
    backend=os.getenv("LLM_AGENT_BACKEND", "ollama"),
    model=os.getenv("LLM_AGENT_MODEL", None),
    endpoint=os.getenv("LLM_AGENT_ENDPOINT", None),
    timeout_s=float(os.getenv("LLM_AGENT_TIMEOUT_S", "120")),
    use_vision=False,
)

if USE_LLM_AGENT:
    if llm_agent.is_configured():
        print("🤖 LLM agent enabled")
        print("Backend:", llm_agent.backend)
        print("Vision Model:", llm_agent.vision_model)
        print("Reasoning Model:", llm_agent.reasoning_model)
    else:
        print("⚠️ LLM agent configuration incomplete")

# ---------------- MAIN EXECUTION ----------------
def main():
    """Main autonomous execution loop"""
    print("🚀 Starting Autonomous Robot Reflection Experiment")
    print(f"Task Context: {TASK_CONTEXT}")
    print("=" * 60)
    
    # Spawn random objects
    objects = spawn_random_objects(MAX_OBJECTS)
    print(f"📦 Spawned {len(objects)} objects")
    
    # Execution state
    state = "scan"
    target_object = None
    goal_position = None
    detected_objects = []
    attempt_count = 0
    successful_tasks = 0
    
    # Policy parameters for LLM learning
    policy = {
        "x_offset": 0.0,
        "y_offset": 0.0,
        "grasp_height": 0.03,
        "approach_height": 0.10,
        "lift_height": 0.15,
        "release_delay": 30,
    }
    
    attempt_history = []
    
    while p.isConnected() and attempt_count < 20:  # Max 20 attempts
        if state == "scan":
            print(f"\n🔍 Scan cycle {attempt_count + 1}")
            
            # Detect and select target
            target_object, detected_objects = detect_and_select_target()
            
            if target_object is None:
                print("No more objects to handle. Task complete!")
                break
            
            # Plan placement
            goal_position = plan_intelligent_placement(target_object, detected_objects)
            
            # Update global cube position for compatibility
            global cube
            cube = target_object['body_id']
            
            state = "plan"
            attempt_count += 1
            
        elif state == "plan":
            print(f"\n📋 Planning approach to {target_object['type']}")
            
            # Get current object position
            cube_pos, _ = p.getBasePositionAndOrientation(cube)
            cube_pos = np.array(cube_pos)
            
            # Apply learned policy offsets
            perceived_cube_pos = cube_pos.copy()
            perceived_cube_pos[0] += policy["x_offset"]
            perceived_cube_pos[1] += policy["y_offset"]
            
            # Plan approach
            approach_target = perceived_cube_pos.copy()
            approach_target[2] += policy["approach_height"]
            
            grasp_target = perceived_cube_pos.copy()
            grasp_target[2] += policy["grasp_height"]
            
            state = "approach"
            
        elif state == "approach":
            print("🚗 Approaching object...")
            smooth_move(approach_target)
            state = "descend"
            
        elif state == "descend":
            print("⬇️ Descending to grasp...")
            smooth_move(grasp_target)
            state = "grasp"
            
        elif state == "grasp":
            print("🤏 Attempting grasp...")
            grip_pos = p.getLinkState(robot, ee_index)[0]
            cube_pos, _ = p.getBasePositionAndOrientation(cube)
            
            dist = np.linalg.norm(np.array(grip_pos) - np.array(cube_pos))
            
            if dist < 0.08:
                # Successful grasp
                relative_offset = [
                    cube_pos[0] - grip_pos[0],
                    cube_pos[1] - grip_pos[1],
                    cube_pos[2] - grip_pos[2],
                ]
                
                constraint_id = p.createConstraint(
                    gripper,
                    -1,
                    cube,
                    -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0],
                    relative_offset,
                )
                
                p.changeConstraint(constraint_id, maxForce=300)
                print("✅ Object grasped successfully!")
                state = "lift"
                
            else:
                print("❌ Grasp failed - triggering reflection...")
                last_failure_type = "grasp_failure"
                state = "reflect"
                
        elif state == "lift":
            print("⬆️ Lifting object...")
            lift_target = cube_pos.copy()
            lift_target[2] += policy["lift_height"]
            smooth_move(lift_target)
            state = "place_above"
            
        elif state == "place_above":
            print("🎯 Moving to placement location...")
            above_goal = goal_position.copy()
            above_goal[2] += policy["approach_height"]
            smooth_move(above_goal)
            state = "lower_to_place"
            
        elif state == "lower_to_place":
            print("⬇️ Lowering to place...")
            place_target = goal_position.copy()
            place_target[2] += 0.025
            smooth_move(place_target)
            state = "release"
            
        elif state == "release":
            print("🤚 Releasing object...")
            time.sleep(policy["release_delay"] / 1000)  # Convert ms to seconds
            
            # Release constraint
            constraints = p.getNumConstraints()
            for i in range(constraints):
                constraint_info = p.getConstraintInfo(i)
                if constraint_info[2] == cube or constraint_info[1] == cube:
                    p.removeConstraint(i)
                    break
            
            # Update placement history
            navigator.update_placement_history(goal_position)
            
            print("✅ Object placed successfully!")
            successful_tasks += 1
            state = "scan"
            
        elif state == "reflect":
            print("\n🧠 LLM Reflection Analysis")
            print("=" * 40)
            
            # Get error information
            error, rgb = get_relative_pixel_error_overhead_and_rgb(
                target_body_id=cube,
                reference_body_id=gripper,
                verbose=False,
            )
            
            pixel_error_x = 0.0
            pixel_error_y = 0.0
            cube_visible = error is not None
            
            if error is None:
                print("Object not visible - limited observations")
            else:
                pixel_error_x, pixel_error_y = error
            
            # Scene info for LLM
            scene_info = {
                "failure_type": last_failure_type,
                "retry_count": attempt_count,
                "cube_visible": bool(cube_visible),
                "pixel_error_x": float(pixel_error_x),
                "pixel_error_y": float(pixel_error_y),
                "target_object_type": target_object['type'],
                "task_context": TASK_CONTEXT,
            }
            
            # Get LLM decision
            decision = llm_agent.reflect(
                scene_info=scene_info,
                policy=policy,
                rgb=rgb,
                history=attempt_history,
            )
            
            print(f"💡 LLM Analysis: {decision.explanation}")
            print(f"🔧 Suggested updates: {decision.updates}")
            
            # Apply policy updates
            policy = apply_policy_updates(policy, decision.updates)
            print(f"📊 Updated policy: {policy}")
            
            # Record attempt
            attempt_history.append({
                "attempt": attempt_count,
                "failure_type": last_failure_type,
                "target_type": target_object['type'],
                "updates": decision.updates,
                "mode": decision.mode,
            })
            
            if decision.terminate:
                print("🛑 LLM requested termination")
                break
            
            state = "plan"
        
        p.stepSimulation()
        time.sleep(1/240)
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 AUTONOMOUS TASK SUMMARY")
    print("=" * 60)
    print(f"Total Attempts: {attempt_count}")
    print(f"Successful Tasks: {successful_tasks}")
    print(f"Success Rate: {(successful_tasks/attempt_count*100):.1f}%" if attempt_count > 0 else "N/A")
    print(f"Task Context: {TASK_CONTEXT}")
    print(f"Final Policy: {policy}")
    
    if attempt_history:
        print(f"\n🧠 Learning Summary:")
        print(f"LLM Reflections: {len(attempt_history)}")
        print(f"Policy Improvements: {sum(1 for h in attempt_history if h['updates'])}")

if __name__ == "__main__":
    main()
