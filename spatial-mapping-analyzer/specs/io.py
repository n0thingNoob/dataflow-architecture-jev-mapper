import hashlib
import json
from pathlib import Path

import yaml


def read_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_yaml(path: Path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def fingerprint(model):
    content = json.dumps(model.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()
