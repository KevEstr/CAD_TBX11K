import time
import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class WSConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self.active[session_id] = ws
        logger.info("WS connected: %s", session_id)

    async def send_progress(self, session_id: str, current: int, total: int, detail: str):
        ws = self.active.get(session_id)
        if ws is None:
            return
        try:
            await ws.send_json({
                "type": "progress",
                "current": current,
                "total": total,
                "percent": round(current / total * 100, 1),
                "detail": detail,
            })
        except Exception:
            self.disconnect(session_id)

    async def send_complete(self, session_id: str, heatmap: list, overlay: str, time_ms: float):
        ws = self.active.get(session_id)
        if ws is None:
            return
        try:
            await ws.send_json({
                "type": "complete",
                "occlusion_heatmap_6x6": heatmap,
                "occlusion_overlay_base64": overlay,
                "occlusion_time_ms": time_ms,
            })
        except Exception:
            pass
        finally:
            self.disconnect(session_id)

    async def send_error(self, session_id: str, message: str):
        ws = self.active.get(session_id)
        if ws is None:
            return
        try:
            await ws.send_json({"type": "error", "detail": message})
            await ws.close()
        except Exception:
            pass
        finally:
            self.disconnect(session_id)

    def disconnect(self, session_id: str):
        self.active.pop(session_id, None)
        logger.info("WS disconnected: %s", session_id)