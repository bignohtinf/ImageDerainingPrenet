import json
import os

def save_training_history(save_dir, history):
    """Save training history to JSON file"""
    history_path = os.path.join(save_dir, 'training_history.json')
    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save training history: {e}")