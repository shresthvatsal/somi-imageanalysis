import cv2
import numpy as np
from PIL import Image


def enhance_lines(image_path, output_path=None, show=False):
    """
    Enhance lines using bilateral filter and contrast boosting.
    Optionally save the result or show it directly.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equ = cv2.equalizeHist(gray)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(equ, -1, kernel)

    # Convert back to BGR
    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    if output_path:
        cv2.imwrite(output_path, result)

    if show:
        Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).show()

    return result


def denoise_image(image_path, output_path=None, show=False):
    """
    Reduce noise while preserving edges using bilateral filtering.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    denoised = cv2.bilateralFilter(img, 9, 75, 75)

    if output_path:
        cv2.imwrite(output_path, denoised)

    if show:
        Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)).show()

    return denoised
