from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FrameSample:
    timestamp_seconds: float
    image_bytes: bytes


@dataclass
class VideoWindow:
    start_seconds: float
    end_seconds: float
    frames: list[FrameSample]


@dataclass
class WindowResult:
    start_seconds: float
    end_seconds: float
    event_type: str
    summary: str
    observed_cues: list[str] = field(default_factory=list)
    needs_review: bool = True
    review_status: str = "unreviewed"
    clip_name: str | None = None
    warning: str | None = None
    summary_embedding: list[float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowResult":
        """Safely construct WindowResult ignoring unexpected fields."""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class AnalysisSession:
    session_id: str
    source_name: str
    duration_seconds: float
    model_name: str
    events: list[WindowResult]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisSession":
        """Safely construct AnalysisSession from serialized dictionary."""
        raw_events = data.get("events", [])
        events = [WindowResult.from_dict(ev) if isinstance(ev, dict) else ev for ev in raw_events]
        return cls(
            session_id=data.get("session_id", ""),
            source_name=data.get("source_name", ""),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            model_name=data.get("model_name", ""),
            events=events,
            warnings=data.get("warnings", []),
        )
