import json

def load_config():
    cfg = {}
    with open("bag_files.json", "r") as f:
        cfg = json.load(f)
    return cfg    