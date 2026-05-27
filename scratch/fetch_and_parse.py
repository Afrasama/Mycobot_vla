import urllib.request
import json
import subprocess
import time
import sys

p = subprocess.Popen([sys.executable, "run_dashboard.py"])
time.sleep(2)

try:
    url = "http://localhost:8080/api/logs?_t=" + str(int(time.time() * 1000))
    print("Fetching:", url)
    with urllib.request.urlopen(url) as resp:
        body = resp.read().decode("utf-8")
        print("Response length:", len(body))
        print("Is valid JSON?", end=" ")
        try:
            data = json.loads(body)
            print("YES! Count:", len(data))
            if len(data) > 0:
                print("First log session ID:", data[0].get("session_id"))
        except Exception as je:
            print("NO! Error:", je)
            print("Body prefix:", body[:500])
except Exception as e:
    print("Connection error:", e)
finally:
    p.terminate()
    p.wait()
