import sys
from pathlib import Path

print(Path(__file__).resolve().parents[1])
sys.path.append(str(Path(__file__).resolve().parents[1]))
