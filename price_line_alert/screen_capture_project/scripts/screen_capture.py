import pyautogui
from datetime import datetime
import os
import time

def capture_chart_region(save_dir):
    # Wait for 1 minute (you can reduce for testing)
    print("⏳ Waiting 60 seconds for you to open the browser and position the chart...")
    time.sleep(60)

    # Define the region to capture: (left, top, width, height)
    # You must update these values based on your screen and where the chart is
    region = (100, 200, 800, 600)  # <-- CHANGE THIS

    screenshot = pyautogui.screenshot(region=region)

    # Save the screenshot with a timestamp
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{save_dir}/snippet_{timestamp}.png"
    screenshot.save(filepath)
    print(f"✅ Captured chart region: {filepath}")

# For testing
if __name__ == "__main__":
    capture_chart_region("images/")
