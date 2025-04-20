from screen_capture import capture_chart_region
from analyze_image import detect_intersections
from send_alert import send_email_alert

import os

SAVE_DIR = "screenshots/"
INTERVAL = 3  # Time in minutes
ALERT_EMAIL = "recipient_email@gmail.com"

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Step 1: Capture screen
    capture_chart_region(SAVE_DIR)


    # Step 2: Analyze screenshots periodically
    for image in os.listdir(SAVE_DIR):
        intersections = detect_intersections(os.path.join(SAVE_DIR, image))
        if intersections:
            send_email_alert(ALERT_EMAIL, "Intersection Detected", f"Found intersections in {image}")
