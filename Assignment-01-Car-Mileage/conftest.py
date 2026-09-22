"""Pytest bootstrap: make the project root importable (src, predict).

With this file at the project root, pytest adds the root directory to
sys.path automatically, so `from src import config` and `import predict`
work whether tests are run via `python -m pytest` or a plain `pytest`.
"""
