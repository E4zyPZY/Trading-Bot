"""Test config with new PCA settings."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config

cfg = load_config(Path('.'))
print('✓ Config loaded')
print(f'  - Model type: {cfg["model"].get("type", "N/A")}')
print(f'  - PCA components: {cfg["model"]["pca_components"]}')
print(f'  - Long threshold: {cfg["model"]["long_threshold"]}')
print(f'  - Short threshold: {cfg["model"]["short_threshold"]}')
print('\n✓ All config parameters ready for main.py')
