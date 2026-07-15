import os


def ensure_dirs():
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('static', exist_ok=True)
