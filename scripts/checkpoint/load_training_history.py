import json
import os

def load_training_history(save_dir):
    """Load training history from JSON file"""
    history_path = os.path.join(save_dir, 'training_history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            print(f"Loaded training history with {len(history)} epochs")
            return history
        except Exception as e:
            print(f"Warning: Could not load training history: {e}")
            return []
    return []