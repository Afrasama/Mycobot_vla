import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

# Automatically load environment variables from .env
import utils.env_loader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURE_LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "failure_log.jsonl")
EXECUTION_LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "execution.log")

def setup_execution_logger():
    """Setup comprehensive execution logger"""
    os.makedirs(os.path.dirname(EXECUTION_LOG_PATH), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('robot_execution')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(os.path.dirname(EXECUTION_LOG_PATH), f"execution_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log session start
    logger.info("=" * 80)
    logger.info("ROBOT EXECUTION SESSION STARTED")
    logger.info("=" * 80)
    
    return logger, log_file


def log_failure(
    failure_type: str,
    robot_state: Dict[str, Any],
    llm_response: Any,
    strategy_chosen: str,
) -> None:
    """
    Append a structured failure/recovery record to the JSONL log and insert it into SQLite.
    """
    os.makedirs(os.path.dirname(FAILURE_LOG_PATH), exist_ok=True)

    session_id = os.getenv("ROBOT_SESSION_ID", "session_legacy")

    # Format LLM response to ensure we capture explanation and adjustments properly
    llm_reasoning = {}
    if hasattr(llm_response, "dict") and callable(llm_response.dict):
        llm_reasoning = llm_response.dict()
    elif hasattr(llm_response, "__dict__"):
        llm_reasoning = llm_response.__dict__
    elif isinstance(llm_response, dict):
        llm_reasoning = llm_response
    else:
        llm_reasoning = {"explanation": str(llm_response), "adjustments": {}}

    # Parse attempt number from robot_state or default to 1
    attempt = 1
    if isinstance(robot_state, dict):
        attempt = robot_state.get("attempt")
        if attempt is None:
            scene_info = robot_state.get("scene_info", {})
            if isinstance(scene_info, dict):
                attempt = scene_info.get("attempt")
                if attempt is None and "retry_count" in scene_info:
                    attempt = int(scene_info["retry_count"]) + 1
    if attempt is None:
        attempt = 1

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "failure_type": failure_type,
        "attempt": attempt,
        "robot_state": robot_state,
        "llm_response": llm_response,
        "strategy_chosen": strategy_chosen,
    }

    # 1. Write to global log backup (JSONL)
    with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")

    # 2. Write to session-specific log file (JSONL)
    if session_id != "session_legacy":
        session_dir = os.path.join(os.path.dirname(FAILURE_LOG_PATH), "sessions")
        os.makedirs(session_dir, exist_ok=True)
        session_log_path = os.path.join(session_dir, f"failure_log_{session_id}.jsonl")
        with open(session_log_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")

    # 3. Insert into MongoDB if available
    try:
        from pymongo import MongoClient
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        # Set a short 1.5s timeout so the simulation doesn't hang if MongoDB is offline
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1500)
        db = client["mycobot_db"]
        collection = db["simulation_logs"]
        
        # Save record
        mongo_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "failure_type": failure_type,
            "attempt": attempt,
            "robot_state": robot_state,
            "llm_reasoning": llm_reasoning,
            "strategy_chosen": strategy_chosen
        }
        collection.insert_one(mongo_record)
        print("[DATABASE] Successfully saved log to MongoDB!")
    except Exception as e:
        # Graceful warning if MongoDB is offline
        print(f"[DATABASE WARNING] MongoDB write skipped: {e}. Saving to SQLite local cache.")

    # 4. Insert into SQLite Database
    try:
        import sqlite3
        db_path = os.path.join(os.path.dirname(FAILURE_LOG_PATH), "simulation_logs.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_id TEXT,
                failure_type TEXT,
                attempt INTEGER,
                robot_state TEXT,
                llm_reasoning TEXT,
                strategy_chosen TEXT
            )
        """)
        
        # Insert record
        cursor.execute("""
            INSERT INTO session_logs (timestamp, session_id, failure_type, attempt, robot_state, llm_reasoning, strategy_chosen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            session_id,
            failure_type,
            attempt,
            json.dumps(robot_state),
            json.dumps(llm_reasoning),
            strategy_chosen
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DATABASE ERROR] Could not save log to SQLite: {e}")
        print(f"[DATABASE ERROR] Could not save log to SQLite: {e}")

def log_robot_state(logger, state: str, details: str = "", attempt: int = 0, distance: float = None):
    """Log robot state changes"""
    distance_str = f"{distance:.3f}m" if distance is not None else "N/A"
    message = f"STATE: {state} | {details} | Attempt: {attempt} | Distance: {distance_str}"
    logger.info(message)

def log_llm_decision(logger, decision):
    """Log LLM decision details"""
    logger.info("=" * 60)
    logger.info("LLM DECISION")
    logger.info("=" * 60)
    logger.info(f"Mode: {decision.mode}")
    logger.info(
        f"Confidence: {decision.confidence:.3f}"
        if decision.confidence is not None
        else "Confidence: N/A"
    )
    logger.info(f"Explanation: {decision.explanation}")
    logger.info(f"Updates: {decision.updates}")
    logger.info("=" * 60)

def log_policy_update(logger, old_policy, new_policy):
    """Log policy changes"""
    logger.info("POLICY UPDATE")
    logger.info(f"Old: {old_policy}")
    logger.info(f"New: {new_policy}")
    logger.info("=" * 40)

def log_session_summary(
    logger,
    total_attempts: int,
    final_distance: float,
    success: bool,
    *,
    grasp_success: bool = False,
    gripper_model: str = "",
    failure_type: str = "",
):
    """Log session summary with explicit pick vs place outcomes."""
    logger.info("=" * 80)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Attempts: {total_attempts}")
    logger.info(f"Final Distance (cube–goal): {final_distance:.3f}m")
    logger.info(f"Place succeeded (stable at goal): {'YES' if success else 'NO'}")
    logger.info(f"Pick succeeded (contacts + lift test): {'YES' if grasp_success else 'NO'}")
    if gripper_model:
        logger.info(f"Gripper URDF: {gripper_model}")
    if failure_type:
        logger.info(f"Last failure / exit note: {failure_type}")
    logger.info("=" * 80)
    logger.info("ROBOT EXECUTION SESSION ENDED")
    logger.info("=" * 80)
