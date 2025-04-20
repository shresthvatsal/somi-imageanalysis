import cv2
import numpy as np
from itertools import combinations

def detect_intersections(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Line detection via Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=50, maxLineGap=20)

    intersections = []

    def get_line_params(line):
        x1, y1, x2, y2 = line[0]
        return np.array([x1, y1]), np.array([x2, y2])

    def compute_intersection(p1, p2, p3, p4):
        """Compute the intersection point of line segments p1-p2 and p3-p4 if it exists."""
        def ccw(a, b, c):
            return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

        if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
            A = np.array([[p2[0] - p1[0], p3[0] - p4[0]],
                          [p2[1] - p1[1], p3[1] - p4[1]]])
            b = np.array([p3[0] - p1[0], p3[1] - p1[1]])
            try:
                t, s = np.linalg.solve(A, b)
                intersection = p1 + t * (p2 - p1)
                return tuple(intersection.astype(int))
            except np.linalg.LinAlgError:
                return None
        return None

    if lines is not None:
        print(f"Found {len(lines)} lines in {image_path}")
        for line1, line2 in combinations(lines, 2):
            p1, p2 = get_line_params(line1)
            p3, p4 = get_line_params(line2)
            intersect = compute_intersection(p1, p2, p3, p4)
            if intersect:
                intersections.append(intersect)

        print(f"Found {len(intersections)} intersections.")
        return intersections
    else:
        print(f"No lines found in {image_path}")
        return []
