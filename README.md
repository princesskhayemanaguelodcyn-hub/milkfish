## Bangus Freshness Detection (Eye Color + EM Clustering)

A minimal FastAPI service that estimates bangus (milkfish) freshness from an eye image using Expectation Maximization (Gaussian Mixture) clustering on HSV color pixels in the eye region, plus simple heuristics.

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open the interactive docs at `http://localhost:8000/docs`.

### API

- `GET /health` — health check
- `POST /classify` — form-data upload with field `file` (image). Optional query `components` (2-5, default 3).

Example with curl:

```bash
curl -X POST "http://localhost:8000/classify?components=3" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/bangus_eye.jpg" | jq .
```

Example response:

```json
{
  "success": true,
  "label": "fresh",
  "confidence": 0.83,
  "metrics": {
    "whiteness_ratio": 0.71,
    "yellowness_ratio": 0.09,
    "darkness_ratio": 0.12,
    "sharpness_score": 0.46
  },
  "eye_detected": true,
  "eye_circle": { "x": 412, "y": 231, "r": 38 },
  "used_fallback_region": false,
  "cluster_centers_hsv": [[0.10, 0.12, 0.88], [0.12, 0.46, 0.62], [0.05, 0.09, 0.18]],
  "cluster_counts": [1342, 612, 221]
}
```

### Method overview

- Resize image for efficiency, detect eye via Hough Circles with a contour fallback.
- Extract HSV pixels inside the detected circular eye region.
- Run EM (Gaussian Mixture) clustering over HSV pixels.
- Compute ratios for whiteness (low S, high V), yellowness (H≈20°–60°, high S), darkness (low V).
- Combine with a simple sharpness estimate (Laplacian variance) to classify into: fresh, moderate fresh, spoiled, or rotten.

Notes:
- This is a heuristic baseline and not a substitute for a model trained on labeled data.
- Results are sensitive to lighting and framing; close-up eye images under consistent lighting work best.

License: MIT