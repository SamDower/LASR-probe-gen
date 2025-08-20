from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data"


class SmartPath:
    def __init__(self, base_path):
        self.path = base_path
        self.path.mkdir(parents=True, exist_ok=True)

    def __truediv__(self, other):
        return str(self.path / other)

    def __str__(self):
        return str(self.path)


class data:
    data = SmartPath(DATA_DIR)
    refusal = SmartPath(DATA_DIR / "refusal")
    # new_behaviour = SmartPath(DATA_DIR / "new_behaviour")
