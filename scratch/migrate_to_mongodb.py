import sqlite3
import json
import os
from pymongo import MongoClient

db_path = "logs/simulation_logs.db"
mongo_uri = "mongodb://localhost:27017/"

if not os.path.exists(db_path):
    print("No SQLite database found to migrate.")
    exit(0)

try:
    # Connect to MongoDB
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
    db = client["mycobot_db"]
    collection = db["simulation_logs"]
    
    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, session_id, failure_type, attempt, robot_state, llm_reasoning, strategy_chosen 
        FROM session_logs
    """)
    rows = cursor.fetchall()
    
    migrated_count = 0
    for row in rows:
        timestamp, session_id, failure_type, attempt, robot_state_str, llm_reasoning_str, strategy_chosen = row
        
        try:
            robot_state = json.loads(robot_state_str)
        except Exception:
            robot_state = {}
            
        try:
            llm_reasoning = json.loads(llm_reasoning_str)
        except Exception:
            llm_reasoning = {}
            
        # Check if record already exists in MongoDB to prevent duplicates
        existing = collection.find_one({
            "timestamp": timestamp,
            "session_id": session_id,
            "attempt": attempt
        })
        
        if not existing:
            mongo_record = {
                "timestamp": timestamp,
                "session_id": session_id,
                "failure_type": failure_type,
                "attempt": attempt,
                "robot_state": robot_state,
                "llm_reasoning": llm_reasoning,
                "strategy_chosen": strategy_chosen
            }
            collection.insert_one(mongo_record)
            migrated_count += 1
            
    print(f"Migration completed! Successfully migrated {migrated_count} records to MongoDB.")
    conn.close()
    
except Exception as e:
    print(f"Migration failed: {e}")
