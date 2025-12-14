from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"


class SmartPath:
    def __init__(self, base_path):
        self.path = Path(base_path)
        if not self.path.suffix:  # Don't mkdir if this is a file
            self.path.mkdir(parents=True, exist_ok=True)

    def __truediv__(self, other):
        new_path = self.path / other
        if not new_path.suffix:  # Create directory only if this is not a file
            new_path.mkdir(parents=True, exist_ok=True)
        return SmartPath(new_path)

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return f"SmartPath({self.path})"

    def __fspath__(self):
        return str(self.path)

    def as_posix(self):
        return self.path.as_posix()

    def joinpath(self, *args):
        return self.__truediv__(Path(*args))

    def exists(self):
        return self.path.exists()

    def is_file(self):
        return self.path.is_file()

    def is_dir(self):
        return self.path.is_dir()

    def open(self, *args, **kwargs):
        return self.path.open(*args, **kwargs)

class data:
    probe_gen = SmartPath(REPO_ROOT / "src" / "probe_gen")
    data = SmartPath(DATA_DIR)
    figures = SmartPath(DATA_DIR / "figures")

    refusal = SmartPath(DATA_DIR / "refusal")
    lists = SmartPath(DATA_DIR / "lists")
    metaphors = SmartPath(DATA_DIR / "metaphors")
    science = SmartPath(DATA_DIR / "science")
    sycophancy = SmartPath(DATA_DIR / "sycophancy")
    authority = SmartPath(DATA_DIR / "authority")
    deception = SmartPath(DATA_DIR / "deception")
    sandbagging = SmartPath(DATA_DIR / "sandbagging")
    bias = SmartPath(DATA_DIR / "bias")