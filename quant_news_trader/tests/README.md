# Test Layout

This folder is organized by test scope:

- `unit/`: Fast checks for individual components and local smoke tests.
- `integration/`: Tests that validate interactions across modules.
- `system/`: End-to-end or subsystem validation scripts.

## Run all tests

```bash
python -m pytest tests/
```

## Run by scope

```bash
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/system/
```

## Run script-style validation tests

Some system checks are script-like and can also be run directly:

```bash
python tests/system/test_risk_modules.py
python tests/system/test_advanced_systems.py
python tests/integration/test_final_integration.py
```
