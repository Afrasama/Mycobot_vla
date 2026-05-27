import os

def load_env():
    """
    Robust, zero-dependency helper to load key-value pairs from .env file into os.environ.
    Finds the .env file in the project root directory.
    """
    # Locate project root relative to this file
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    env_path = os.path.join(project_root, ".env")
    
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Inline comments (custom loader does not use python-dotenv)
                    if " #" in value:
                        value = value.split(" #", 1)[0].strip()
                    elif value.startswith("#"):
                        value = ""
                    # Strip surrounding quotes if present
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    os.environ[key] = value
        return True
    return False

# Automatically invoke load_env upon importing this module to ensure env values are ready
load_env()
