import cv2
import numpy as np

def extract_card_corners(mask, epsilon_factor=0.02):
    """
    Extracts a 4-point contour from a binary mask using contour approximation.

    Args:
        mask (np.ndarray): Binary mask (e.g., from SAM), dtype=uint8, shape=(H, W)
        epsilon_factor (float): Approximation precision; smaller = tighter fit

    Returns:
        corners (np.ndarray): 4 points (top-left, top-right, bottom-right, bottom-left), shape=(4, 2)
    """
    # Ensure mask is binary
    mask = (mask > 0).astype("uint8") * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Approximate polygon
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 4:
        raise ValueError(f"Expected 4-point contour, got {len(approx)} points")

    corners = approx[:, 0].astype("float32")  # Shape (4, 2)

    # Order the corners consistently: TL, TR, BR, BL
    return order_card_corners(corners)

def order_card_corners(pts):
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect
