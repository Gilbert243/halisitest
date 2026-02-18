import yaml

class EvalConfig:
    def __init__(self, path: str):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        self.api = cfg["api"]
        self.paths = cfg["paths"]
        self.evaluation = cfg["evaluation"]
