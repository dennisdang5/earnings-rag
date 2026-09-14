from pathlib import Path
import yaml

def load_questions(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        return yaml.safe_load(f)