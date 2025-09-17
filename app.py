import os
from typing import Any, Dict, Tuple

import cv2
import gradio as gr
import numpy as np

from detector import classify_freshness, compute_freshness_metrics, crop_eye_roi, detect_eye_circle


def analyze(image: np.ndarray) -> Tuple[np.ndarray, str, Dict[str, Any], str]:
	"""Analyze an uploaded RGB image and return annotated image, label, metrics, and explanation."""
	if image is None or image.size == 0:
		return None, "No image provided.", {}, "Please upload a photo of the fish eye."

	# Convert RGB (from Gradio) to BGR (for OpenCV)
	image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	# Detect eye circle
	circle = detect_eye_circle(image_bgr)
	annotated = image_bgr.copy()
	roi = image_bgr
	eye_detected = False

	if circle is not None:
		eye_detected = True
		cx, cy, r = circle
		cv2.circle(annotated, (cx, cy), r, (0, 255, 0), 2)
		cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), 3)
		roi = crop_eye_roi(image_bgr, circle)
	else:
		# Fallback to a central crop if eye not found
		h, w = image_bgr.shape[:2]
		size = max(32, min(h, w) // 3)
		cy, cx = h // 2, w // 2
		y1, y2 = max(0, cy - size), min(h, cy + size)
		x1, x2 = max(0, cx - size), min(w, cx + size)
		roi = image_bgr[y1:y2, x1:x2]

	metrics = compute_freshness_metrics(roi)
	label, score, explanation = classify_freshness(metrics)
	if not eye_detected:
		explanation = "Eye not confidently detected. Used central crop. " + explanation

	# Prepare outputs
	annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
	result_text = f"{label} (score {score:.2f})"
	metrics_rounded: Dict[str, Any] = {k: round(float(v), 4) for k, v in metrics.items()}
	metrics_rounded["eye_detected"] = bool(eye_detected)

	return annotated_rgb, result_text, metrics_rounded, explanation


with gr.Blocks(title="Bangus Freshness by Eye") as demo:
	gr.Markdown("""
	**Bangus (Milkfish) Freshness Detector**

	Upload a clear photo of the fish head focusing on the eye. The app detects the eye, analyzes clarity/whiteness/contrast, and classifies freshness.
	""")

	with gr.Row():
		input_image = gr.Image(label="Upload bangus eye photo", type="numpy")
		output_image = gr.Image(label="Annotated image")

	output_label = gr.Textbox(label="Freshness")
	output_metrics = gr.JSON(label="Metrics")
	output_expl = gr.Markdown()

	analyze_btn = gr.Button("Analyze")
	analyze_btn.click(
		fn=analyze,
		inputs=[input_image],
		outputs=[output_image, output_label, output_metrics, output_expl],
	)

	gr.Examples(
		examples=[],
		inputs=input_image,
	)


if __name__ == "__main__":
	server_name = os.environ.get("HOST", "0.0.0.0")
	server_port = int(os.environ.get("PORT", "7860"))
	demo.launch(server_name=server_name, server_port=server_port)

