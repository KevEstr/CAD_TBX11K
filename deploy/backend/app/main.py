import uuid
import time
import json
import logging
import threading
import asyncio
import tempfile
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse

from app.config import (
    MODEL_PATH, MANIFEST_PATH, FEAT_COLS_PATH,
    MAX_UPLOAD_SIZE, ALLOWED_EXTENSIONS, SESSION_TTL, CLASS_NAMES,
)
from app.inference import InferencePipeline
from app.explainer import RadiographExplainer
from app.ws_manager import WSConnectionManager

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
ws_manager = WSConnectionManager()
session_store: dict = {}
session_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, explainer
    logger.info("Loading model...")
    try:
        pipeline = InferencePipeline()
        with open(FEAT_COLS_PATH) as f:
            feat_cols = json.load(f)
        explainer = RadiographExplainer(pipeline.pipeline, feat_cols, pipeline.selected_names)
        explainer.set_session_store(session_store)
        logger.info("Model loaded successfully")
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


def _create_session() -> tuple[str, Path]:
    session_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.mkdtemp(prefix=f"session_{session_id}_"))
    with session_lock:
        session_store[session_id] = {"temp_dir": temp_dir, "created_at": time.time()}
    return session_id, temp_dir


def _cleanup_session(session_id: str):
    with session_lock:
        entry = session_store.pop(session_id, None)
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
    return {
        "model": "XGBoost",
        "classes": CLASS_NAMES,
        "selector": pipeline.manifest.get("selector"),
        "imbalance": pipeline.manifest.get("estrategia"),
        "features_selected": len(pipeline.selected_names),
        "features_total": len(pipeline.feat_cols_3300),
        "selected_model": pipeline.manifest.get("seleccionado_cv"),
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
        result = pipeline.predict(contents, str(temp_dir))
    except Exception as e:
        _cleanup_session(session_id)
        raise HTTPException(500, detail=f"Prediction failed: {str(e)}")

    pred_class_id = result["pred_class_id"]
    proba = result["probabilities"]
    img_256 = result["img_preprocessed"]

    try:
        level1 = explainer.explain_level1(result["vec_2000"], pred_class_id, img_256)
    except Exception as e:
        logger.warning("Level1 failed: %s", e)
        level1 = None

    try:
        level2 = explainer.explain_level2(result["vec_2000"], pred_class_id, img_256)
    except Exception as e:
        logger.warning("Level2 failed: %s", e)
        level2 = None

    elapsed = (time.time() - t0) * 1000

    response = {
        "session_id": session_id,
        "prediction": {
            "class": result["pred_class"],
            "class_id": int(pred_class_id),
            "confidence": round(float(proba[CLASS_NAMES[pred_class_id]]), 4),
        },
        "probabilities": {k: round(float(v), 4) for k, v in proba.items()},
        "level1": level1,
        "level2": level2,
        "metadata": {
            "model": "XGBoost",
            "selector": f"MI_N{len(pipeline.selected_names)}",
            "imbalance": "class_weight",
            "preprocessing": "BF(d=9,s=80) + CLAHE(clip=40) + lung_seg",
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


# --- WebSocket ---
@app.websocket("/ws/{session_id}")
async def websocket_occlusion(ws: WebSocket, session_id: str):
    with session_lock:
        entry = session_store.get(session_id)
    if entry is None:
        await ws.accept()
        await ws.send_json({"type": "error", "detail": "Session not found"})
        await ws.close()
        return

    await ws_manager.connect(session_id, ws)

    try:
        temp_dir = str(entry["temp_dir"])
        main_loop = asyncio.get_event_loop()
        result = await main_loop.run_in_executor(
            None,
            explainer.run_occlusion_sync,
            temp_dir, session_id, ws_manager, main_loop,
        )
        if result:
            heatmap, overlay_b64, elapsed = result
            await ws_manager.send_complete(session_id, heatmap, overlay_b64, elapsed)
    except WebSocketDisconnect:
        logger.info("WS disconnected during occlusion: %s", session_id)
    except Exception as e:
        logger.error("Occlusion failed: %s", e)
        try:
            await ws_manager.send_error(session_id, f"Occlusion failed: {str(e)}")
        except Exception:
            pass
    finally:
        _cleanup_session(session_id)
        _cleanup_session(session_id)


# --- Serve frontend static files ---
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"

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