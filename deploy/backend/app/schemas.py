from pydantic import BaseModel
from typing import Optional

class PredictResponse(BaseModel):
    session_id: str
    prediction: dict
    probabilities: dict
    level1: dict
    level2: dict
    metadata: dict

class ErrorResponse(BaseModel):
    error: str
    detail: str

class WsProgress(BaseModel):
    type: str = "progress"
    current: int
    total: int
    percent: float
    detail: str

class WsComplete(BaseModel):
    type: str = "complete"
    occlusion_heatmap_6x6: list
    occlusion_overlay_base64: str
    occlusion_time_ms: float

class WsError(BaseModel):
    type: str = "error"
    detail: str