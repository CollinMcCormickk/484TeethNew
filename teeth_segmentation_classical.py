import cv2
import numpy as np


def get_feature_detector():
    """Return SIFT if available, otherwise fallback to ORB."""
    if hasattr(cv2, "SIFT_create"):
        try:
            return cv2.SIFT_create()
        except Exception:
            pass

    if hasattr(cv2, "ORB_create"):
        return cv2.ORB_create(nfeatures=500)

    raise RuntimeError("No supported OpenCV feature detector found (SIFT or ORB required).")


def generate_classical_mask(image, size=(256, 256), min_area_ratio=0.0005, max_area_ratio=0.1):
    """Create a candidate teeth mask using CLAHE, thresholding, and contour filtering."""
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(image)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    mask = np.zeros_like(closed, dtype=np.uint8)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = image.shape[0] * image.shape[1]
    min_area = max(20, int(area * min_area_ratio))
    max_area = int(area * max_area_ratio)

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if min_area < contour_area < max_area:
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return (mask.astype(np.float32) / 255.0)


def feature_keypoint_heatmap(image, size=(256, 256), detector=None, blur_size=(9, 9)):
    """Return a normalized keypoint density heatmap for the image."""
    if detector is None:
        detector = get_feature_detector()

    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    keypoints = detector.detect(image, None)

    heatmap = np.zeros(size, dtype=np.float32)
    height, width = heatmap.shape

    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] += 1.0

    if np.max(heatmap) > 0:
        heatmap = heatmap / heatmap.max()

    if blur_size[0] % 2 == 0:
        blur_size = (blur_size[0] + 1, blur_size[1])
    if blur_size[1] % 2 == 0:
        blur_size = (blur_size[0], blur_size[1] + 1)

    heatmap = cv2.GaussianBlur(heatmap, blur_size, 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def build_hybrid_input(image, size=(256, 256), feature_mode="hybrid", detector=None):
    """Compose input channels from radiograph, classical mask, and keypoint heatmap."""
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image_float = image.astype(np.float32) / 255.0
    classical_mask = generate_classical_mask(image, size=size)

    if feature_mode == "classical":
        return np.stack([image_float, classical_mask], axis=0)

    if feature_mode in {"keypoint", "sift", "orb"}:
        heatmap = feature_keypoint_heatmap(image, size=size, detector=detector)
        return np.stack([image_float, heatmap], axis=0)

    heatmap = feature_keypoint_heatmap(image, size=size, detector=detector)
    return np.stack([image_float, classical_mask, heatmap], axis=0)
