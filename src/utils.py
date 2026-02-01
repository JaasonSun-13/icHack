"""Utility helpers: logging, config loading, pickle I/O."""
import logging, os, pickle, yaml

def setup_logging(level="INFO"):
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
