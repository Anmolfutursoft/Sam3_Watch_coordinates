from pydantic import BaseModel


class SegmentationResponse(BaseModel):
    success: bool
    message: str
    bbox: list[int] | None = None