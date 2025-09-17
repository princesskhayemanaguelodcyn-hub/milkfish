from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


def resize_preserving_aspect_ratio(image_bgr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Resize image so that the longest side is at most max_side."""
    height, width = image_bgr.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return image_bgr
    scale = max_side / float(long_side)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    return cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)


def detect_eye_circle(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int]]]:
    """Detect a circular eye using HoughCircles. Returns mask and (x, y, r) or (None, None)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    height, width = gray.shape[:2]
    min_dim = min(height, width)
    min_radius = max(8, int(0.03 * min_dim))
    max_radius = max(min_radius + 2, int(0.15 * min_dim))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(0.15 * min_dim),
        param1=100,
        param2=20,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, thickness=-1)
        return mask, (int(x), int(y), int(r))

    # Fallback: choose a dark, circular-ish area via threshold + contour
    thresh_val = int(np.clip(np.mean(gray) * 0.7, 20, 120))
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.medianBlur(thresh, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * (area / (perim * perim))
        if circularity < 0.5:
            continue
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        radius_est = 0.5 * (w_ + h_) / 2
        score = circularity * area
        if score > best_score:
            best_score = score
            best = (int(x_ + w_ / 2), int(y_ + h_ / 2), int(radius_est))
    if best is not None:
        mask = np.zeros((height, width), dtype=np.uint8)
        xc, yc, r = best
        r = max(5, int(r))
        cv2.circle(mask, (int(xc), int(yc)), int(r), 255, thickness=-1)
        return mask, (int(xc), int(yc), int(r))

    return None, None


def get_center_crop_mask(image_bgr: np.ndarray, fraction: float = 0.3) -> np.ndarray:
    """Fallback mask: centered circular mask covering a fraction of min dimension."""
    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = int(fraction * min(height, width) / 2)
    cx, cy = width // 2, height // 2
    cv2.circle(mask, (cx, cy), max(5, radius), 255, thickness=-1)
    return mask


def extract_hsv_pixels(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return HSV pixels inside mask as float32 in [0, 1]."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    pixels = hsv[mask > 0]
    return pixels


def compute_image_sharpness_laplacian(image_bgr: np.ndarray) -> float:
    """Estimate sharpness via variance of Laplacian (normalized)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip(variance / 500.0, 0.0, 1.0))


essential_min_pixels = 30


def run_em_on_hsv(pixels_hsv: np.ndarray, n_components: int = 3, random_state: int = 42) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Fit GMM on HSV pixels and return (model, labels, centers)."""
    min_needed = max(50, n_components * 20)
    if pixels_hsv.shape[0] < min_needed:
        # Too few pixels for a stable GMM; tile and clip to stabilize
        reps = int(np.ceil(min_needed / max(1, pixels_hsv.shape[0])))
        pixels_hsv = np.clip(np.vstack([pixels_hsv] * reps), 0.0, 1.0)
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
    gmm.fit(pixels_hsv)
    labels = gmm.predict(pixels_hsv)
    centers = gmm.means_
    return gmm, labels, centers


def compute_color_ratios_from_clusters(pixels_hsv: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> Dict[str, float]:
    """Compute whiteness, yellowness, darkness ratios using cluster-wise heuristics and pixel fallback."""
    total = float(len(labels)) if len(labels) > 0 else 1.0
    cluster_counts = np.bincount(labels, minlength=centers.shape[0]).astype(np.float32)
    cluster_ratios = cluster_counts / total

    whiteness = 0.0
    yellowness = 0.0
    darkness = 0.0

    for idx, center in enumerate(centers):
        h, s, v = center.tolist()
        ratio = float(cluster_ratios[idx])
        is_white_like = (s < 0.22) and (v > 0.65)
        is_yellow_like = (0.06 <= h <= 0.17) and (s > 0.25) and (v > 0.4)
        is_dark_like = (v < 0.35)
        if is_white_like:
            whiteness += ratio
        if is_yellow_like:
            yellowness += ratio
        if is_dark_like:
            darkness += ratio

    # Pixel-level supplement in case clusters miss rare colors
    h = pixels_hsv[:, 0]
    s = pixels_hsv[:, 1]
    v = pixels_hsv[:, 2]
    whiteness_px = np.mean(((s < 0.22) & (v > 0.65)).astype(np.float32))
    yellowness_px = np.mean(((h >= 0.06) & (h <= 0.17) & (s > 0.25) & (v > 0.4)).astype(np.float32))
    darkness_px = np.mean((v < 0.35).astype(np.float32))

    # Blend cluster and pixel-level stats
    whiteness = float(0.7 * whiteness + 0.3 * whiteness_px)
    yellowness = float(0.7 * yellowness + 0.3 * yellowness_px)
    darkness = float(0.7 * darkness + 0.3 * darkness_px)

    return {
        "whiteness_ratio": float(np.clip(whiteness, 0.0, 1.0)),
        "yellowness_ratio": float(np.clip(yellowness, 0.0, 1.0)),
        "darkness_ratio": float(np.clip(darkness, 0.0, 1.0)),
    }


def classify_freshness(color_scores: Dict[str, float], sharpness_score: float) -> Tuple[str, float, str]:
    """Map color metrics and sharpness to freshness labels with a rough confidence."""
    w = color_scores["whiteness_ratio"]
    y = color_scores["yellowness_ratio"]
    d = color_scores["darkness_ratio"]

    # Rule-based mapping informed by domain heuristics
    if (w > 0.65 and y < 0.15 and d < 0.20) or (w > 0.55 and sharpness_score > 0.35 and y < 0.18):
        label = "fresh"
    elif (y >= 0.35 and d < 0.45) or (w < 0.35 and y >= 0.25):
        label = "spoiled"
    elif (d >= 0.45) or (w < 0.20 and y > 0.40):
        label = "rotten"
    else:
        label = "moderate fresh"

    # Confidence: coarse distance from decision boundaries
    conf = 0.5
    if label == "fresh":
        conf = 0.5 + 0.5 * float(min(1.0, (w - max(y, d)) / 0.6))
    elif label == "moderate fresh":
        conf = 0.35 + 0.3 * float(np.clip(1.0 - abs(w - 0.5), 0.0, 1.0))
    elif label == "spoiled":
        conf = 0.5 + 0.5 * float(min(1.0, (max(y, d) - w) / 0.6))
    elif label == "rotten":
        conf = 0.6 + 0.4 * float(min(1.0, (d - max(w, 1 - y)) / 0.6))

    rationale = f"w={w:.2f}, y={y:.2f}, d={d:.2f}, sharp={sharpness_score:.2f}"
    return label, float(np.clip(conf, 0.0, 1.0)), rationale


def analyze_image(image_bgr: np.ndarray, components: int = 3) -> Dict:
    """Full pipeline: detect eye, cluster HSV, compute metrics, classify."""
    image_bgr = resize_preserving_aspect_ratio(image_bgr, 1024)
    sharpness = compute_image_sharpness_laplacian(image_bgr)

    mask, circle = detect_eye_circle(image_bgr)
    used_fallback = False
    if mask is None:
        mask = get_center_crop_mask(image_bgr, fraction=0.4)
        used_fallback = True

    pixels_hsv = extract_hsv_pixels(image_bgr, mask)
    if pixels_hsv.shape[0] < essential_min_pixels:
        raise ValueError("Not enough pixels detected in eye region.")

    _, labels, centers = run_em_on_hsv(pixels_hsv, n_components=max(2, min(components, 5)))
    color_scores = compute_color_ratios_from_clusters(pixels_hsv, labels, centers)
    label, confidence, rationale = classify_freshness(color_scores, sharpness)

    cluster_counts = np.bincount(labels, minlength=centers.shape[0]).tolist()
    centers_list = centers.tolist()

    return {
        "label": label,
        "confidence": confidence,
        "metrics": {
            **color_scores,
            "sharpness_score": sharpness,
        },
        "eye_detected": circle is not None,
        "eye_circle": {"x": circle[0], "y": circle[1], "r": circle[2]} if circle else None,
        "used_fallback_region": used_fallback,
        "cluster_centers_hsv": centers_list,
        "cluster_counts": cluster_counts,
    }