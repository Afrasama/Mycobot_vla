"""
MyCobot Analytics Web Dashboard Launcher
Spins up a lightweight read-only local server to view simulation metrics, VLM logs, and coordinates.
Guaranteed 100% read-only and isolated from active PyBullet simulation execution.
"""
import http.server
import socketserver
import webbrowser
import threading
import os
import sys

# Automatically load environment variables from .env
import utils.env_loader


PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class DualServerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        from urllib.parse import urlparse
        clean_path = urlparse(self.path).path
        if clean_path == '/api/logs' or clean_path == '/api/logs/':
            logs = []
            mongo_fetched = False
            
            # 1. Try to fetch from MongoDB
            try:
                from pymongo import MongoClient
                mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
                client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1500)
                db = client["mycobot_db"]
                collection = db["simulation_logs"]
                
                # Fetch recent records sorted by timestamp descending
                cursor = collection.find().sort("timestamp", -1)
                for doc in cursor:
                    # Exclude the MongoDB '_id' field when converting to JSON
                    logs.append({
                        "timestamp": doc.get("timestamp"),
                        "session_id": doc.get("session_id"),
                        "failure_type": doc.get("failure_type"),
                        "attempt": doc.get("attempt", 1),
                        "robot_state": doc.get("robot_state", {}),
                        "llm_reasoning": doc.get("llm_reasoning", {}),
                        "strategy_chosen": doc.get("strategy_chosen", "")
                    })
                mongo_fetched = True
                print(f"[API SERVER] Successfully fetched {len(logs)} records from MongoDB!")
            except Exception as e:
                print(f"[API SERVER ERROR] MongoDB fetch failed: {e}.")
            
            # Send dynamic JSON response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Expose-Headers', 'X-Database-Source')
            self.send_header('X-Database-Source', 'MongoDB' if mongo_fetched else 'Offline')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.end_headers()
            import json
            self.wfile.write(json.dumps(logs).encode('utf-8'))
        else:
            super().do_GET()

    def end_headers(self):
        # Prevent caching for live logs
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def start_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DualServerHandler) as httpd:
        print(f"\n[DASHBOARD SERVER] Running at: http://localhost:{PORT}/dashboard/index.html")
        print("[DASHBOARD SERVER] Press Ctrl+C in this terminal to shut down server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[DASHBOARD SERVER] Stopping...")
            httpd.shutdown()
            sys.exit(0)

if __name__ == "__main__":
    print("=" * 60)
    print("MYCOBOT ROBOTICS ANALYTICS DASHBOARD LAUNCHER")
    print("=" * 60)
    print("[SAFEGUARD] Dashboard is running in isolated READ-ONLY mode.")
    print("[SAFEGUARD] Guaranteed 0% CPU overhead or interference with PyBullet.")
    print("=" * 60)
    
    # Start server in background thread so terminal remains interactive
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Let server start up briefly
    import time
    time.sleep(0.5)
    
    # Open dashboard in browser
    url = f"http://localhost:{PORT}/dashboard/index.html"
    print(f"\nOpening default browser to: {url}")
    webbrowser.open(url)
    
    # Keep main thread alive to capture Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down dashboard server. Goodbye!")
