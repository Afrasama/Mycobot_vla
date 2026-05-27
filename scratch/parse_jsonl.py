import json
import os

log_path = r"g:\Afrasama\mycobot_dumps\ollama_mycobot_grippertesting\logs\failure_log.jsonl"

if not os.path.exists(log_path):
    print("Logs path not found.")
    exit()

print("Scanning logs for LLM Reflection outputs...")
count = 0
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            # Find a record where the LLM successfully responded with an explanation
            llm_response = data.get("llm_response", {})
            explanation = llm_response.get("explanation", "")
            updates = llm_response.get("updates", {})
            failure_type = data.get("failure_type", "")
            
            if explanation and "timed out" not in explanation.lower() and "placed_successfully" not in failure_type:
                count += 1
                print(f"\n--- SUCCESSFUL LLM REFLECTION EXAMPLE {count} ---")
                print(f"Recorded Failure Type: {failure_type}")
                print(f"LLM Explanation:\n{explanation}")
                print(f"Proposed Updates: {updates}")
                
                if count >= 3:
                    break
        except Exception as e:
            continue

if count == 0:
    print("No non-timeout LLM reflections found in the JSONL file.")
