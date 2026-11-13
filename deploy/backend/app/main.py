import uuid
import time
import json
import logging
import threading
import asyncio
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse

from app.config import (
    MODEL_PATH, MANIFEST_PATH, SCHEMA_PATH,
    MAX_UPLOAD_SIZE, ALLOWED_EXTENSIONS, SESSION_TTL, CLASS_NAMES,
)
from app.inference import InferencePipeline
from app.explainer import RadiographExplainer
from src.lung_segmentation import warmup_segmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return super().default(o)

# --- Global state ---
pipeline: InferencePipeline = None
explainer: RadiographExplainer = None
session_store: dict = {}
session_lock = threading.Lock()

# Occlusion state lives here (module level so the background thread and the
# polling endpoints share it). All entries are pruned by _background_gc and
# _cleanup_session to avoid unbounded growth.
occlusion_progress: dict = {}
occlusion_results: dict = {}
occlusion_task_started: set = set()
occlusion_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="occlusion")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, explainer
    logger.info("Loading model...")
    try:
        pipeline = InferencePipeline()
        explainer = RadiographExplainer(
            pipeline.pipeline, pipeline.schema["feature_columns"]
        )
        logger.info(
            "Model loaded successfully (%s)",
            pipeline.manifest.get("seleccionado_cv"),
        )
        logger.info("Warming up segmentation model...")
        try:
            warmup_segmentation()
            logger.info("Segmentation model ready")
        except Exception as e:
            logger.warning("Segmentation warmup failed: %s", e)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise
    yield
    logger.info("Shutting down")


app = FastAPI(title="CAD-TBX11K", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_image(filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail="Only PNG, JPG, JPEG allowed")
    return ext


def _encode_png(img) -> str:
    import base64 as _b64
    import cv2 as _cv2
    ok, buf = _cv2.imencode(".png", img)
    if not ok:
        return ""
    return "data:image/png;base64," + _b64.b64encode(buf.tobytes()).decode()


def _encode_segmentation(img, mask) -> str:
    import base64 as _b64
    import cv2 as _cv2
    import numpy as np

    if img is None or mask is None:
        return ""

    base = _cv2.cvtColor(img, _cv2.COLOR_GRAY2BGR)
    mask_bool = np.asarray(mask, dtype=bool)
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    tint = base.copy()
    tint[mask_bool] = (196, 211, 94)
    out = _cv2.addWeighted(tint, 0.18, base, 0.82, 0)
    contours, _ = _cv2.findContours(mask_u8, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    _cv2.drawContours(out, contours, -1, (196, 211, 94), 2)

    ok, buf = _cv2.imencode(".png", out)
    if not ok:
        return ""
    return "data:image/png;base64," + _b64.b64encode(buf.tobytes()).decode()


def _encode_segmentation_from_original(original_path: str, mask512) -> str:
    import cv2 as _cv2
    import numpy as np

    if mask512 is None:
        return ""
    img = _cv2.imread(original_path, _cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""

    h, w = img.shape[:2]
    scale = min(1.0, 1024.0 / max(h, w))
    if scale < 1.0:
        img = _cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=_cv2.INTER_AREA,
        )

    out_h, out_w = img.shape[:2]
    mask = _cv2.resize(
        np.asarray(mask512, dtype=np.uint8),
        (out_w, out_h),
        interpolation=_cv2.INTER_LINEAR,
    ) > 0
    return _encode_segmentation(img, mask)


def _create_session() -> tuple[str, Path]:
    session_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.mkdtemp(prefix=f"session_{session_id}_"))
    with session_lock:
        session_store[session_id] = {"temp_dir": temp_dir, "created_at": time.time()}
    return session_id, temp_dir


def _cleanup_session(session_id: str):
    with session_lock:
        entry = session_store.pop(session_id, None)
    occlusion_task_started.discard(session_id)
    occlusion_progress.pop(session_id, None)
    occlusion_results.pop(session_id, None)
    if entry:
        temp_dir = entry["temp_dir"]
        try:
            shutil.rmtree(str(temp_dir), ignore_errors=True)
        except Exception:
            pass
        logger.info("Session cleaned up: %s", session_id)


def _background_gc():
    while True:
        time.sleep(30)
        now = time.time()
        with session_lock:
            stale = [sid for sid, e in list(session_store.items())
                     if now - e["created_at"] > SESSION_TTL]
            for sid in stale:
                entry = session_store.pop(sid, None)
                if entry:
                    shutil.rmtree(str(entry["temp_dir"]), ignore_errors=True)
                    logger.info("GC cleaned stale session: %s", sid)
                occlusion_task_started.discard(sid)
                occlusion_progress.pop(sid, None)
                occlusion_results.pop(sid, None)
            # Drop occlusion state whose session no longer exists (abandoned
            # clients that never DELETE, or crashed jobs).
            valid = set(session_store.keys())
            for sid in list(occlusion_progress.keys()):
                if sid not in valid:
                    occlusion_progress.pop(sid, None)
                    occlusion_results.pop(sid, None)
                    occlusion_task_started.discard(sid)


gc_thread = threading.Thread(target=_background_gc, daemon=True)


@app.on_event("startup")
async def _start_gc():
    gc_thread.start()


# --- REST Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.get("/api/v1/model/info")
async def model_info():
    if pipeline is None:
        raise HTTPException(503, "Model not loaded")
    m = pipeline.manifest
    return {
        "model": m.get("seleccionado_cv"),
        "classes": CLASS_NAMES,
        "feature_set": "textura + HOG",
        "features_input": len(pipeline.feat_cols),
        "features_selected": m.get("selector", {}).get("n_features"),
        "threshold": pipeline.threshold,
        "note": "Prototipo de investigación; sin validación externa.",
    }


def _run_prediction(contents: bytes, temp_dir: str) -> dict:
    """Trabajo CPU-bound (prediccion + explicacion). Se ejecuta fuera del loop."""
    result = pipeline.predict(contents, temp_dir)
    proba = result["probabilities"]
    t0 = time.time()
    try:
        level1 = explainer.explain_level1(result["vec"], result["img_thorax"])
    except Exception as e:
        logger.warning("Level1 failed: %s", e)
        level1 = None
    return {
        "result": result,
        "proba": proba,
        "level1": level1,
        "explain_ms": (time.time() - t0) * 1000,
    }


@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(503, "Model not loaded")
    if explainer is None:
        raise HTTPException(503, "Explainer not loaded")

    _validate_image(file.filename)
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, detail="File too large. Max 20MB")

    session_id, temp_dir = _create_session()
    t0 = time.time()

    try:
        out = await asyncio.to_thread(_run_prediction, contents, str(temp_dir))
    except Exception as e:
        _cleanup_session(session_id)
        raise HTTPException(500, detail=f"Prediction failed: {str(e)}")

    result = out["result"]
    proba = out["proba"]
    level1 = out["level1"]
    elapsed = (time.time() - t0) * 1000

    response = {
        "session_id": session_id,
        "prediction": {
            "class": result["pred_class"],
            "class_id": int(result["pred_class_id"]),
            "confidence": round(float(max(proba.values())), 4),
            "tb_probability": round(float(proba["tb"]), 4),
            "tb_positive": bool(result["tb_positive"]),
            "threshold": result["threshold"],
        },
        "probabilities": {k: round(float(v), 4) for k, v in proba.items()},
        "probabilities_raw": {k: round(float(v), 4) for k, v in result["probabilities_raw"].items()},
        "image_base64": _encode_png(result["img_thorax"]),
        "segmentation_base64": _encode_segmentation_from_original(result["original_path"], result.get("mask512")),
        "level1": level1,
        "metadata": {
            "model": "Logistic Regression (notebook 08, 3 clases)",
            "features": "1.832: textura interior + HOG",
            "preprocessing": "PSPNet + recorte toracico + BF + CLAHE", 
            "inference_time_ms": round(elapsed, 1),
        },
    }
    return json.loads(json.dumps(response, cls=NumpyEncoder))


@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    _cleanup_session(session_id)
    return {"status": "cleaned"}


@app.get("/api/v1/session/{session_id}")
async def get_session_status(session_id: str):
    with session_lock:
        exists = session_id in session_store
    return {"exists": exists}


# --- Occlusion Progress & Polling ---
@app.post("/api/v1/predict/{session_id}/occlude")
async def start_occlusion(session_id: str):
    """Start occlusion processing in background. Returns immediately."""
    with session_lock:
        entry = session_store.get(session_id)
    if entry is None:
        raise HTTPException(404, detail="Session not found")

    if session_id in occlusion_task_started:
        return {"status": "already_started"}

    occlusion_task_started.add(session_id)
    occlusion_progress[session_id] = {"current": 0, "total": 36, "detail": "Starting..."}

    temp_dir = str(entry["temp_dir"])
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        occlusion_executor,
        _run_occlusion_background,
        temp_dir, session_id,
    )
    return {"status": "started"}


def _run_occlusion_background(temp_dir: str, session_id: str):
    """Run occlusion in background thread, writing progress to shared dicts."""
    try:
        heatmap, overlay_b64, elapsed = explainer.run_occlusion_sync_polling(
            temp_dir, session_id, occlusion_progress
        )
        occlusion_results[session_id] = {
            "heatmap": heatmap,
            "overlay": overlay_b64,
            "time_ms": round(elapsed * 1000, 1),
            "status": "complete",
        }
        occlusion_progress[session_id] = {"current": 36, "total": 36, "percent": 100.0, "detail": "Complete"}
    except Exception as e:
        occlusion_task_started.discard(session_id)
        if "Session cancelled" in str(e):
            occlusion_progress.pop(session_id, None)
            occlusion_results.pop(session_id, None)
            logger.info("Occlusion cancelled: %s", session_id)
        else:
            occlusion_results[session_id] = {"status": "error", "detail": str(e)}
            logger.error("Occlusion background failed: %s", e)


@app.get("/api/v1/predict/{session_id}/progress")
async def get_occlusion_progress(session_id: str):
    progress = occlusion_progress.get(session_id)
    resp = {}
    if progress is None:
        resp = {"status": "not_started", "current": 0, "total": 36}
    else:
        result = occlusion_results.get(session_id)
        if result:
            resp = {"status": "complete", "result": result}
        else:
            resp = {"status": "processing", "progress": progress}
    return json.loads(json.dumps(resp, cls=NumpyEncoder))


# --- Serve frontend static files ---
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/")
async def serve_frontend():
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>CAD-TBX11K API</h1><p>Frontend not found. Use /docs for API.</p>")

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    fp = FRONTEND_DIR / file_path
    if fp.exists() and fp.is_file():
        ct = {"css": "text/css", "js": "application/javascript", "png": "image/png", "ico": "image/x-icon", "jpg": "image/jpeg"}
        ext = fp.suffix[1:]
        return HTMLResponse(fp.read_bytes(), media_type=ct.get(ext, "text/plain"))
    return HTMLResponse("", status_code=404)