from pydantic import BaseModel


class SegmentationResponse(BaseModel):
    success: bool
    message: str
    bbox: list[int] | None = None
    boxes_list: list[list[list[float]]] | None = None
    scores_list: list[list[float]] | None = None
    