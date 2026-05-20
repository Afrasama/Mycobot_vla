import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import log_failure

def test_session_logging():
    # Simulate a running session
    session_id = "session_20260519_testing123"
    os.environ["ROBOT_SESSION_ID"] = session_id
    
    print(f"Setting ROBOT_SESSION_ID = {session_id}")
    
    dummy_state = {
        "joint_angles": [0.11, -0.22, 0.33, -0.44, 0.55, -0.66],
        "scene_info": {
            "pixel_error_x": 14.5,
            "pixel_error_y": -9.2,
            "distance_to_goal": 0.187
        }
    }
    
    dummy_response = {
        "explanation": "Gripper claw closed too early during pre-grasp phase.",
        "updates": {"grasp_height": -0.005, "release_delay": 45}
    }
    
    print("Logging simulated failure...")
    log_failure(
        failure_type="robust_pick_failure",
        robot_state=dummy_state,
        llm_response=dummy_response,
        strategy_chosen="retry_with_policy_update"
    )
    
    # Confirm output files
    global_log = os.path.join("data", "failure_log.jsonl")
    session_log = os.path.join("data", "sessions", f"failure_log_{session_id}.jsonl")
    
    print("\nVerification Results:")
    print(f"Global log exists: {os.path.exists(global_log)}")
    print(f"Session log exists: {os.path.exists(session_log)}")
    
    if os.path.exists(session_log):
        with open(session_log, "r", encoding="utf-8") as f:
            content = f.read().strip()
            print(f"Logged record content:\n{content}")

if __name__ == "__main__":
    test_session_logging()
