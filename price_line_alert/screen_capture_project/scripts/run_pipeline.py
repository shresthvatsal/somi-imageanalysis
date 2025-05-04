import os
import time
from datetime import datetime

from screen_capture import capture_browser_area
from image_cleaning_utils import enhance_lines
from image_analysis_utils import detect_line_intersections_improved
from send_alert import send_email_alert

# Configuration
IMAGE_NAME_PREFIX = "chart_capture"
SAVE_DIR = os.path.join("screenshots", "captures")
ALERT_EMAIL = "vatsalshresth@gmail.com"  # Destination email for alerts

os.makedirs(SAVE_DIR, exist_ok=True)


def run_pipeline():
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[+] ({timestamp}) Capturing chart area...")

        # Step 1: Capture the chart and save it
        image_path = os.path.join(SAVE_DIR, f"{IMAGE_NAME_PREFIX}_{timestamp}.png")
        capture_browser_area(image_path)

        # Step 2: Enhance / clean the captured image
        cleaned_path = image_path.replace(".png", "_cleaned.png")
        enhance_lines(image_path, cleaned_path)

        # Step 3: Detect line intersections in the cleaned image using the improved function
        print("[+] Detecting intersections...")
        intersections = detect_line_intersections_improved(cleaned_path, debug=False)

        # Check if there are intersections between purple & red or purple & green
        if intersections["Purple-Red"] or intersections["Purple-Green"]:
            send_email_alert(
                ALERT_EMAIL,
                "Intersection Detected",
                f"Found intersections in {os.path.basename(image_path)}"
            )
        else:
            print("[-] No significant intersections detected.")

        print("[✓] Pipeline complete.\n")
        # Sleep briefly before starting the next iteration.
        time.sleep(10)


if __name__ == "__main__":
    run_pipeline()
