"""Test config with new PCA settings."""
from src.config_loader import load_config
from pathlib import Path

cfg = load_config(Path('.'))
print('✓ Config loaded')
print(f'  - Model type: {cfg["model"].get("type", "N/A")}')
print(f'  - PCA components: {cfg["model"]["pca_components"]}')
print(f'  - Long threshold: {cfg["model"]["long_threshold"]}')
print(f'  - Short threshold: {cfg["model"]["short_threshold"]}')
print('\n✓ All config parameters ready for main.py')
