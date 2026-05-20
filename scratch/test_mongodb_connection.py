import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import log_failure
from pymongo import MongoClient

def test_connection():
    print("Testing MongoDB Integration...")
    
    # 1. Verify environment variable is loaded
    mongo_uri = os.getenv("MONGODB_URI")
    print(f"Loaded MONGODB_URI: {mongo_uri}")
    if not mongo_uri:
        print("ERROR: MONGODB_URI is not set in the environment!")
        return False
        
    # 2. Connect directly to verify accessibility
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        db = client["mycobot_db"]
        collection = db["simulation_logs"]
        print("Successfully connected to MongoDB server!")
    except Exception as e:
        print(f"ERROR: Direct MongoDB connection failed: {e}")
        return False

    # 3. Trigger standard log_failure logic to write a record
    test_session_id = f"test_verification_{int(datetime.now(timezone.utc).timestamp())}"
    print(f"Logging dummy failure to session: {test_session_id}")
    
    dummy_robot_state = {
        "attempt": 1,
        "joint_angles": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "scene_info": {
            "distance_to_goal": 0.05,
            "pixel_error_x": 0.0,
            "pixel_error_y": 0.0
        }
    }
    
    dummy_llm_response = {
        "explanation": "Verification run test",
        "updates": {"x_offset": 0.0, "y_offset": 0.0}
    }
    
    # Set ROBOT_SESSION_ID env var so logger picks it up
    os.environ["ROBOT_SESSION_ID"] = test_session_id
    
    try:
        log_failure(
            failure_type="verification_test",
            robot_state=dummy_robot_state,
            llm_response=dummy_llm_response,
            strategy_chosen="test_strategy"
        )
    except Exception as e:
        print(f"ERROR: log_failure function raised an exception: {e}")
        return False
        
    # 4. Fetch the record back from MongoDB and verify
    print("Searching MongoDB for the logged verification record...")
    try:
        record = collection.find_one({"session_id": test_session_id})
        if record:
            print("SUCCESS! Found the record in MongoDB:")
            print(f"  - Session ID: {record.get('session_id')}")
            print(f"  - Failure Type: {record.get('failure_type')}")
            print(f"  - Strategy Chosen: {record.get('strategy_chosen')}")
            print(f"  - LLM Explanation: {record.get('llm_reasoning', {}).get('explanation')}")
            
            # Clean up the test record
            collection.delete_one({"session_id": test_session_id})
            print("Successfully cleaned up test record from MongoDB.")
            return True
        else:
            print("ERROR: Did not find the test record in MongoDB simulation_logs collection.")
            return False
    except Exception as e:
        print(f"ERROR: Failed to read from MongoDB: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
