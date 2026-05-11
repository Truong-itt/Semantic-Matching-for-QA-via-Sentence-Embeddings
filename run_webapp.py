"""
run_webapp.py
-------------
Launcher script for the Sentence Selection QA web demo.

Usage (from sentence_selection_qa/ directory):
    python run_webapp.py

Then open: http://127.0.0.1:5000
"""

import os
import sys

# Ensure the project root is on the Python path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Auto-install Flask if missing
try:
    import flask
except ImportError:
    import subprocess
    print("Flask not found — installing …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])

from webapp.app import app, get_encoder

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Sentence Selection QA — Web Demo")
    print("  http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    print("Initialising encoder (first run may take ~10 seconds) …")
    get_encoder()
    print("\nServer starting … Press Ctrl+C to quit.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
