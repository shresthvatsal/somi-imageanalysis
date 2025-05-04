import pyautogui
from datetime import datetime
import os
import time

def capture_browser_area(filepath):
    print("⏳ Waiting 60 seconds for browser to be positioned on chart...")
    time.sleep(60)

    # Define the region to capture: (left, top, width, height)
    region = (100, 300, 1900, 600)  # ⬅️ Update these as per your screen and chart

    screenshot = pyautogui.screenshot(region=region)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    screenshot.save(filepath)

    print(f"✅ Screenshot saved: {filepath}")
    return filepath