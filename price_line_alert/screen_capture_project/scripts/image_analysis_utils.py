import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def crop_chart_area(input_path, output_path):
    """Crop chart-only region from the full screenshot."""
    image = Image.open(input_path)

    # Coordinates to extract just the chart region (tweak if needed)
    left = 120
    top = 180
    right = 1320
    bottom = 620

    cropped = image.crop((left, top, right, bottom))
    cropped.save(output_path)
    print(f"[INFO] Cropped chart saved to: {output_path}")

def preprocess_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=2)

def find_nearby_points(mask1, mask2, min_dist=5):
    """Detect intersections by proximity of contours instead of pixel-perfect overlap."""
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points1 = np.vstack([cnt.reshape(-1, 2) for cnt in contours1])
    points2 = np.vstack([cnt.reshape(-1, 2) for cnt in contours2])

    nearby = []
    for p1 in points1:
        dist = np.linalg.norm(points2 - p1, axis=1)
        if dist.min() < min_dist:
            nearby.append(tuple(p1))
    return nearby

def detect_line_intersections(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV color ranges
    purple_lower = np.array([110, 40, 40])
    purple_upper = np.array([160, 255, 255])

    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 70, 50])
    red_upper2 = np.array([180, 255, 255])

    green_lower = np.array([40, 40, 40])
    green_upper = np.array([85, 255, 255])

    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, red_lower1, red_upper1),
        cv2.inRange(hsv, red_lower2, red_upper2)
    )
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)

    # Preprocess
    red_mask = preprocess_mask(red_mask)
    green_mask = preprocess_mask(green_mask)
    purple_mask = preprocess_mask(purple_mask)

    # Find intersections
    pr = find_nearby_points(purple_mask, red_mask)
    pg = find_nearby_points(purple_mask, green_mask)
    gr = find_nearby_points(green_mask, red_mask)

    intersections = {
        "Purple-Red": len(pr),
        "Purple-Green": len(pg),
        "Green-Red": len(gr),
    }

    overlay = image.copy()
    for pt in pr:
        cv2.circle(overlay, pt, 4, (255, 0, 0), -1)
    for pt in pg:
        cv2.circle(overlay, pt, 4, (0, 255, 255), -1)
    for pt in gr:
        cv2.circle(overlay, pt, 4, (0, 0, 255), -1)

    if debug:
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        axs[0, 0].imshow(red_mask, cmap='gray')
        axs[0, 0].set_title("Red Mask")

        axs[0, 1].imshow(green_mask, cmap='gray')
        axs[0, 1].set_title("Green Mask")

        axs[1, 0].imshow(purple_mask, cmap='gray')
        axs[1, 0].set_title("Purple Mask")

        axs[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title("Detected Intersections")

        for ax in axs.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return intersections, overlay
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_line_intersections_improved(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Step 1: Fine-tune HSV thresholds ---
    purple_lower = np.array([125, 50, 50])
    purple_upper = np.array([145, 255, 255])
    red_lower1   = np.array([0, 70, 50])
    red_upper1   = np.array([10, 255, 255])
    red_lower2   = np.array([170, 70, 50])
    red_upper2   = np.array([180, 255, 255])
    green_lower  = np.array([40, 40, 40])
    green_upper  = np.array([80, 255, 255])

    # --- Step 2: Create and clean masks using morphological closing ---
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, red_lower1, red_upper1),
        cv2.inRange(hsv, red_lower2, red_upper2)
    )
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    kernel = np.ones((5, 5), np.uint8)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # --- Step 3: Extract line segments via Hough Transform for each mask ---
    def get_hough_lines(mask):
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
        return lines

    purple_lines = get_hough_lines(purple_mask)
    red_lines = get_hough_lines(red_mask)
    green_lines = get_hough_lines(green_mask)

    # --- Step 4: Compute intersections between purple and red/green lines ---
    intersections = {"Purple-Red": [], "Purple-Green": []}

    def compute_intersection(line1, line2):
        # Each line is of the form [x1, y1, x2, y2]
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Lines are parallel.
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        intersection = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return (int(intersection[0]), int(intersection[1]))

    if purple_lines is not None and red_lines is not None:
        for p_line in purple_lines:
            for r_line in red_lines:
                pt = compute_intersection(p_line, r_line)
                if pt is not None:
                    intersections["Purple-Red"].append(pt)

    if purple_lines is not None and green_lines is not None:
        for p_line in purple_lines:
            for g_line in green_lines:
                pt = compute_intersection(p_line, g_line)
                if pt is not None:
                    intersections["Purple-Green"].append(pt)

    # --- Optional: Debug visualization ---
    if debug:
        overlay = image.copy()
        for pt in intersections["Purple-Red"]:
            cv2.circle(overlay, pt, 4, (255, 0, 0), -1)
        for pt in intersections["Purple-Green"]:
            cv2.circle(overlay, pt, 4, (0, 255, 255), -1)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Improved Detected Intersections")
        plt.axis('off')
        plt.show()

    return intersections
