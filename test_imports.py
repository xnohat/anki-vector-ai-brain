#!/usr/bin/env python3
"""Quick test to verify all modules can be imported."""
import sys
sys.path.insert(1, 'src')

print("Testing imports...")

try:
    import owl
    print("✓ owl (YOLOv8 detection)")
except Exception as e:
    print(f"✗ owl: {e}")

try:
    import vectorbot
    print("✓ vectorbot (Vector SDK)")
except Exception as e:
    print(f"✗ vectorbot: {e}")

try:
    import customgpt
    print("✓ customgpt (OpenAI)")
except Exception as e:
    print(f"✗ customgpt: {e}")

try:
    import ui
    print("✓ ui (CustomTkinter)")
except Exception as e:
    print(f"✗ ui: {e}")

try:
    import speechstream
    print("✓ speechstream (Audio)")
except Exception as e:
    print(f"✗ speechstream: {e}")

print("\nAll critical modules imported successfully!")
print("Ready to run the app (after setting API keys and configuring Vector).")

