import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=2)  # Stronger dilation


def find_nearby_points(mask1, mask2, min_dist=5):
    """Detects intersections by proximity of contours instead of pixel-perfect overlap."""
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points1 = np.vstack([cnt.reshape(-1, 2) for cnt in contours1])
    points2 = np.vstack([cnt.reshape(-1, 2) for cnt in contours2])

    nearby = []
    for p1 in points1:
        dist = np.linalg.norm(points2 - p1, axis=1)
        min_dist_idx = np.argmin(dist)
        if dist[min_dist_idx] < min_dist:
            nearby.append(tuple(p1))
    return nearby


def detect_line_intersections(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tuned HSV ranges
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

    # Find intersections using proximity instead of exact overlap
    pr_intersections = find_nearby_points(purple_mask, red_mask)
    pg_intersections = find_nearby_points(purple_mask, green_mask)
    gr_intersections = find_nearby_points(green_mask, red_mask)

    intersections = {
        "Purple-Red": len(pr_intersections),
        "Purple-Green": len(pg_intersections),
        "Green-Red": len(gr_intersections),
    }

    overlay = image.copy()
    for pt in pr_intersections:
        cv2.circle(overlay, pt, 4, (255, 0, 0), -1)
    for pt in pg_intersections:
        cv2.circle(overlay, pt, 4, (0, 255, 255), -1)
    for pt in gr_intersections:
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
results, _ = detect_line_intersections("WhatsApp Image 2025-04-13 at 20.27.20_82b3269b.jpg", debug=True)
print(results)
