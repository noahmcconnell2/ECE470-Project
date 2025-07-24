import json
import os
from datetime import datetime

def get_timestamped_path(log_dir="logs", prefix="ga_history", ext="json"):
    """Generates a timestamped filename in the given log directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{prefix}_{timestamp}.{ext}")

def init_logger(log_dir="logs"):
    """Initializes the log file path with timestamp."""
    return get_timestamped_path(log_dir=log_dir)

def log_generation(history, gen, population, save_path):
    """Appends sorted population data for this generation and writes to disk."""
    gen_data = {
        "gen": gen,
        "population": [
            {
                "rank": i,
                "genome": ind[:],
                "fitness": ind.fitness.values[0]
            } for i, ind in enumerate(sorted(population, key=lambda x: x.fitness.values[0]))
        ]
    }
    history.append(gen_data)

    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)
