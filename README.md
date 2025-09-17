# Bangus (Milkfish) Freshness Detector – Eye-based

This app estimates the freshness of bangus based on the appearance of the eye. It detects the eye region and analyzes clarity, contrast, and cloudiness to classify the fish as Fresh, Borderline, or Not fresh.

## Features

- Eye detection using circular Hough transform
- Metrics: clarity (Laplacian variance), contrast, whiteness (cloudiness proxy), saturation, brightness
- Simple heuristic classifier with an overall score (0–1)
- Gradio web UI for easy image upload and results visualization

## Quickstart

1. Create a Python environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open the URL printed by Gradio and upload a clear photo of the bangus eye.

## Notes and Limitations

- Works best with close-up, in-focus images of the fish head with the eye visible.
- Very small or heavily occluded eyes may not be detected. The app will fall back to analyzing a central crop and will indicate this in the explanation.
- The classifier is heuristic-based and not a medical/food safety authority. Use it as a screening aid.

## Project Structure

```
app.py          # Gradio UI and inference pipeline
detector.py     # Eye detection, metrics, and classification
requirements.txt
README.md
```

