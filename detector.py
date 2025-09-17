import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
	"""Ensure the bounding box stays within image bounds."""
	x1 = max(0, min(int(x1), width - 1))
	y1 = max(0, min(int(y1), height - 1))
	x2 = max(0, min(int(x2), width))
	y2 = max(0, min(int(y2), height))
	return x1, y1, x2, y2


def detect_eye_circle(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int]]:
	"""Detect a circular eye region using Hough Circles and select the best candidate.

	Returns (x, y, r) for the best circle, or None if not found.
	"""
	if image_bgr is None or image_bgr.size == 0:
		return None

	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	# Reduce noise while preserving edges
	gray_blur = cv2.medianBlur(gray, 5)

	height, width = gray_blur.shape[:2]
	min_radius = max(5, int(min(height, width) * 0.03))
	max_radius = max(min_radius + 2, int(min(height, width) * 0.25))

	best_circle: Optional[Tuple[int, int, int]] = None
	best_score: float = -1.0

	# Try a few parameter combinations for robustness
	for dp, param2 in [(1.2, 40), (1.4, 35), (1.2, 30), (1.6, 28)]:
		circles = cv2.HoughCircles(
			gray_blur,
			cv2.HOUGH_GRADIENT,
			dp=dp,
			minDist=max(16, min(height, width) // 6),
			param1=120,
			param2=param2,
			minRadius=min_radius,
			maxRadius=max_radius,
		)
		if circles is None:
			continue

		for cx, cy, r in circles[0]:
			cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(r))
			if r_i <= 0:
				continue

			# Score candidate by center-dark vs ring-bright contrast (pupil vs iris/sclera)
			r_inner = max(1, int(r_i * 0.35))
			r_outer = max(r_inner + 2, int(r_i * 0.9))

			mask_inner = np.zeros_like(gray, dtype=np.uint8)
			cv2.circle(mask_inner, (cx_i, cy_i), r_inner, 255, -1)

			mask_ring = np.zeros_like(gray, dtype=np.uint8)
			cv2.circle(mask_ring, (cx_i, cy_i), r_outer, 255, -1)
			cv2.circle(mask_ring, (cx_i, cy_i), r_inner + 2, 0, -1)

			center_mean = cv2.mean(gray, mask=mask_inner)[0]
			ring_mean = cv2.mean(gray, mask=mask_ring)[0]
			contrast = ring_mean - center_mean

			# Penalize if center is not sufficiently darker than ring
			score = contrast
			if score > best_score:
				best_score = score
				best_circle = (cx_i, cy_i, r_i)

	return best_circle


def crop_eye_roi(image_bgr: np.ndarray, circle: Tuple[int, int, int]) -> np.ndarray:
	"""Crop a square ROI around the detected eye circle with padding."""
	height, width = image_bgr.shape[:2]
	cx, cy, r = circle
	pad = int(r * 1.2)
	x1, y1, x2, y2 = _clip_box(cx - pad, cy - pad, cx + pad, cy + pad, width, height)
	return image_bgr[y1:y2, x1:x2]


def compute_freshness_metrics(eye_roi_bgr: np.ndarray) -> Dict[str, float]:
	"""Compute clarity, whiteness, contrast, and color metrics from an eye ROI."""
	if eye_roi_bgr is None or eye_roi_bgr.size == 0:
		return {
			"clarity": 0.0,
			"laplacian_variance": 0.0,
			"whiteness": 1.0,
			"contrast": 0.0,
			"saturation_mean": 0.0,
			"brightness_mean": 0.0,
		}

	# HSV metrics
	hsv = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2HSV)
	_, s_channel, v_channel = cv2.split(hsv)
	saturation_mean = float(np.mean(s_channel)) / 255.0
	brightness_mean = float(np.mean(v_channel)) / 255.0

	# Whiteness proxy: high V with low S indicates cloudy/opaque areas
	white_mask = (v_channel > 200) & (s_channel < 40)
	whiteness = float(np.mean(white_mask.astype(np.float32)))

	# Clarity/Sharpness via Laplacian variance
	gray_roi = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)
	laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
	laplacian_variance = float(laplacian.var())
	clarity = float(laplacian_variance / (laplacian_variance + 100.0))

	# Global contrast within ROI (std dev of intensity)
	contrast = float(gray_roi.std()) / 255.0

	return {
		"clarity": clarity,
		"laplacian_variance": laplacian_variance,
		"whiteness": whiteness,
		"contrast": contrast,
		"saturation_mean": saturation_mean,
		"brightness_mean": brightness_mean,
	}


def classify_freshness(metrics: Dict[str, float]) -> Tuple[str, float, str]:
	"""Heuristic classification into Fresh / Borderline / Not fresh with a 0-1 score."""
	clarity = float(metrics.get("clarity", 0.0))
	contrast = float(metrics.get("contrast", 0.0))
	whiteness = float(metrics.get("whiteness", 1.0))
	saturation = float(metrics.get("saturation_mean", 0.0))

	# Aggregate score: prioritize clarity and contrast; penalize whiteness
	score = (
		0.45 * clarity +
		0.30 * contrast +
		0.15 * (1.0 - max(0.0, min(1.0, whiteness))) +
		0.10 * saturation
	)
	score = float(max(0.0, min(1.0, score)))

	if score >= 0.62:
		label = "Fresh"
	elif score >= 0.45:
		label = "Borderline"
	else:
		label = "Not fresh"

	explanation = (
		f"clarity={clarity:.2f}, contrast={contrast:.2f}, "
		f"whiteness={whiteness:.2f}, saturation={saturation:.2f}"
	)
	return label, score, explanation

