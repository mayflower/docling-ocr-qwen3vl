import sys
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if PROJECT_SRC.exists():
    sys.path.insert(0, str(PROJECT_SRC))
