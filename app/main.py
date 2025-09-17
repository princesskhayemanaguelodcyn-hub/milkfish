from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

from .eye_em_classifier import analyze_image

app = FastAPI(title="Bangus Freshness API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/classify")
async def classify_bangus_image(
    file: UploadFile = File(..., description="Image containing bangus eye (ideally close-up)"),
    components: int = 3,
) -> dict:
    if components < 2 or components > 5:
        raise HTTPException(status_code=400, detail="components must be in [2, 5]")
    try:
        contents = await file.read()
        data = np.frombuffer(contents, dtype=np.uint8)
        image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Unable to decode image.")
        result = analyze_image(image_bgr, components=components)
        return {"success": True, **result}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to analyze image: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)