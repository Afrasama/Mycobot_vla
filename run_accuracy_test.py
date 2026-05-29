import os
import sys
import subprocess
import time
import glob
import csv

def main():
    print("=========================================================")
    print("🚀 STARTING AUTOMATED ROBOT ACCURACY & ADAPTATION TEST")
    print("=========================================================")
    print("This test runs 1 round of the pick-and-place task in")
    print("Pipeline Evaluation Mode. It injects a deliberate XY")
    print("perception error on attempt 1, allowing you to observe")
    print("the active VLA recovery or LLM reflection loop adapting")
    print("to secure the object and place it at the goal.")
    print("=========================================================\n")

    # Set up environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))
    env["PIPELINE_EVAL_MODE"] = "1"
    env["MAX_ROUNDS"] = "1"
    env["VLA_DEMO_MODE"] = "0"
    
    # We default USE_LLM_AGENT and USE_VLA_RECOVERY to 1 for this test
    if "USE_LLM_AGENT" not in env:
        env["USE_LLM_AGENT"] = "1"
    if "USE_VLA_RECOVERY" not in env:
        env["USE_VLA_RECOVERY"] = "1"

    script_path = os.path.join("experiments", "improved_kinematics_reflection.py")
    python_exe = os.path.join("robo_env", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    print(f"Running simulation via: {python_exe} {script_path}")
    print("Launching PyBullet and Tkinter Monitor GUI...")
    time.sleep(1.5)

    try:
        # Run subprocess and stream stdout
        process = subprocess.Popen(
            [python_exe, script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream stdout in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output, end="")

        process.wait()
        
        if process.returncode == 0:
            print("\n=========================================================")
            print("✅ SIMULATION COMPLETED SUCCESSFULLY!")
            print("=========================================================")
            
            # Read latest generated CSV to print accuracy details
            plots_dir = os.path.join("data", "plots")
            csv_files = glob.glob(os.path.join(plots_dir, "multi_round_evaluation_*.csv"))
            
            if csv_files:
                # Find the latest CSV
                latest_csv = max(csv_files, key=os.path.getmtime)
                print(f"\n📊 Parsed Accuracy Results ({os.path.basename(latest_csv)}):")
                print("-" * 65)
                
                with open(latest_csv, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    rows = list(reader)
                    
                    for r in rows:
                        round_num, attempt_num, final_dist, success = r
                        success_str = "YES" if success.strip().upper() == "TRUE" else "NO"
                        print(f"Round: {round_num:2s} | Attempt: {attempt_num:2s} | Final Distance to Goal: {float(final_dist):.4f}m | Succeeded: {success_str}")
                print("-" * 65)
            else:
                print("\nNo accuracy CSV metrics file was found in data/plots.")
        else:
            print(f"\n❌ Simulation exited with error code: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error executing accuracy test: {e}")

if __name__ == "__main__":
    main()
