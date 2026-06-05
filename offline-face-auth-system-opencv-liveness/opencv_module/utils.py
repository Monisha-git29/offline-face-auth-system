"""
Utility functions for OpenCV face processing module.

Includes visualization helpers to debug alignment and mathematical utilities.
"""

import cv2
import numpy as np
from typing import Tuple


def draw_alignment_debug(
    image: np.ndarray,
    left_eye: Tuple[float, float],
    right_eye: Tuple[float, float],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draws a visual debug overlay showing the eyes, the line connecting them,
    and a horizontal reference line to verify the tilt.

    Args:
        image (np.ndarray): Original input image (BGR).
        left_eye (Tuple[float, float]): (x, y) coordinates of the left eye.
        right_eye (Tuple[float, float]): (x, y) coordinates of the right eye.
        color (Tuple[int, int, int]): BGR color for lines/circles. Default is green.
        thickness (int): Line/circle outline thickness.

    Returns:
        np.ndarray: Annotated image copy.
    """
    debug_img = image.copy()
    
    # Convert points to integers
    le_pt = (int(left_eye[0]), int(left_eye[1]))
    re_pt = (int(right_eye[0]), int(right_eye[1]))
    mid_pt = (int((le_pt[0] + re_pt[0]) / 2), int((le_pt[1] + re_pt[1]) / 2))

    # Draw eye circles
    cv2.circle(debug_img, le_pt, 5, (0, 0, 255), -1)  # Red circle for left eye
    cv2.circle(debug_img, re_pt, 5, (255, 0, 0), -1)  # Blue circle for right eye
    cv2.circle(debug_img, mid_pt, 4, (0, 255, 255), -1)  # Yellow circle for midpoint

    # Draw line connecting eyes
    cv2.line(debug_img, le_pt, re_pt, color, thickness)

    # Draw a horizontal reference line passing through the midpoint
    ref_left = (mid_pt[0] - 50, mid_pt[1])
    ref_right = (mid_pt[0] + 50, mid_pt[1])
    cv2.line(debug_img, ref_left, ref_right, (128, 128, 128), 1, lineType=cv2.LINE_AA)

    # Compute tilt angle and overlay text
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    cv2.putText(
        debug_img,
        f"Tilt: {angle:.2f} deg",
        (le_pt[0] - 20, le_pt[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return debug_img


def create_synthetic_face(
    size: Tuple[int, int] = (400, 400),
    rotation_angle: float = 15.0,
    noise_level: float = 0.0
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Creates a synthetic, stylized face image containing visual eye markers
    tilted at a specified angle. Extremely useful for testing offline pipelines
    without loading large datasets.

    Args:
        size (Tuple[int, int]): Size of synthetic image (width, height).
        rotation_angle (float): Face rotation in degrees (positive is counter-clockwise).
        noise_level (float): Optional salt-and-pepper noise ratio.

    Returns:
        Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
            - Synthetic face image (BGR)
            - Rotated Left eye coordinates (x, y)
            - Rotated Right eye coordinates (x, y)
    """
    width, height = size
    # Start with a neutral background
    img = np.ones((height, width, 3), dtype=np.uint8) * 45  # Sleek dark gray background

    # Face center
    cx, cy = width // 2, height // 2

    # Draw standard unrotated facial elements on a canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 45
    # Head contour
    cv2.ellipse(canvas, (cx, cy), (80, 110), 0, 0, 360, (200, 200, 200), -1)
    # Hair
    cv2.ellipse(canvas, (cx, cy - 80), (85, 45), 0, 0, 360, (20, 20, 20), -1)
    
    # Original eyes coordinates in unrotated frame
    le_x, le_y = cx - 35, cy - 25
    re_x, re_y = cx + 35, cy - 25

    # Draw eyes on canvas
    cv2.circle(canvas, (le_x, le_y), 10, (255, 255, 255), -1)
    cv2.circle(canvas, (le_x, le_y), 4, (120, 50, 20), -1)  # Pupil
    cv2.circle(canvas, (re_x, re_y), 10, (255, 255, 255), -1)
    cv2.circle(canvas, (re_x, re_y), 4, (120, 50, 20), -1)  # Pupil
    
    # Nose
    cv2.fillPoly(canvas, [np.array([[cx, cy - 10], [cx - 10, cy + 15], [cx + 10, cy + 15]], dtype=np.int32)], (150, 150, 150))
    # Mouth (smile)
    cv2.ellipse(canvas, (cx, cy + 45), (35, 15), 0, 0, 180, (50, 50, 180), 3)

    # Now rotate the canvas around the center to create a tilted face
    M = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)
    rotated_img = cv2.warpAffine(canvas, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(45, 45, 45))

    # Calculate the new eye positions under this rotation matrix
    # [x', y'] = [R_00*x + R_01*y + tx, R_10*x + R_11*y + ty]
    def rotate_point(pt):
        px, py = pt
        rx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ry = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        return (float(rx), float(ry))

    left_eye_rot = rotate_point((le_x, le_y))
    right_eye_rot = rotate_point((re_x, re_y))

    # Add Gaussian noise if requested
    if noise_level > 0.0:
        noise = np.random.normal(0, noise_level * 255, rotated_img.shape).astype(np.float32)
        rotated_img = np.clip(rotated_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return rotated_img, left_eye_rot, right_eye_rot
