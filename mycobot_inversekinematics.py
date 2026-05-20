import pybullet as p
import pybullet_data
import time
import os

# ---------------- connect ----------------
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

p.loadURDF("plane.urdf")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(BASE_DIR,"urdf","mycobot_320.urdf")

robot = p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

# ---------------- find end effector ----------------
ee_index=None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot,i)[12].decode()=="link6":
        ee_index=i
        break

print("end effector index:",ee_index)

# ---------------- add simple gripper ----------------
gripper = p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=p.createCollisionShape(
        p.GEOM_BOX,halfExtents=[0.02,0.02,0.01]),
    baseVisualShapeIndex=p.createVisualShape(
        p.GEOM_BOX,halfExtents=[0.02,0.02,0.01],
        rgbaColor=[0.2,0.2,0.2,1])
)

# attach gripper to wrist
p.createConstraint(
    robot,ee_index,
    gripper,-1,
    p.JOINT_FIXED,
    [0,0,0],
    [0,0,0.04],
    [0,0,0]
)

# ---------------- add cube ----------------
cube = p.loadURDF(
    "cube_small.urdf",
    basePosition=[0.30,0.10,0.02]
)

# ---------------- target position ----------------
target_position=[0.30,0.10,0.20]

# visual marker
target_vis=p.createVisualShape(
    p.GEOM_SPHERE,
    radius=0.02,
    rgbaColor=[1,0,0,1]
)
p.createMultiBody(baseVisualShapeIndex=target_vis,
                  basePosition=target_position)

print("moving end effector to:",target_position)

constraint_id=None

# ---------------- main loop ----------------
while p.isConnected():

    # inverse kinematics
    joint_angles=p.calculateInverseKinematics(
        robot,
        ee_index,
        target_position,
        maxNumIterations=200,
        residualThreshold=1e-4
    )

    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot,j)[2]==p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[j],
                force=400,
                positionGain=0.08,
                velocityGain=0.6
            )

    # --------- automatic grasp when touching ---------
    contacts=p.getContactPoints(gripper,cube)

    if constraint_id is None and len(contacts)>0:
        print("contact detected -> grabbing cube")
        constraint_id=p.createConstraint(
            robot,ee_index,
            cube,-1,
            p.JOINT_FIXED,
            [0,0,0],
            [0,0,-0.02],
            [0,0,0]
        )
        
        # Give the arm a next move to execute
        target_position = [0.10, 0.30, 0.30]
        print(f"lifting and moving cube to new position: {target_position}")

    p.stepSimulation()
    time.sleep(1/240)
