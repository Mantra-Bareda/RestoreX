"""
progress.py

Centralized progress reporting for the CLI.
"""

def report_progress(message):
    print(f"[INFO] {message}", flush=True)
