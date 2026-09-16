"""Thread-local sample identity seen by compression wrappers."""
from __future__ import annotations

import threading

_STATE = threading.local()


def set_sample(*, video_id=None, question_id=None, sample_id=None, extra=None) -> None:
    _STATE.video_id = video_id
    _STATE.question_id = question_id
    _STATE.sample_id = sample_id
    _STATE.extra = extra or {}


def clear_sample() -> None:
    for name in ("video_id", "question_id", "sample_id", "extra"):
        if hasattr(_STATE, name):
            delattr(_STATE, name)


def current_key() -> list:
    video_id = getattr(_STATE, "video_id", None)
    question_id = getattr(_STATE, "question_id", None)
    sample_id = getattr(_STATE, "sample_id", None)
    if video_id is not None and question_id is not None:
        return [str(video_id), str(question_id)]
    if video_id is not None:
        return [str(video_id)]
    if sample_id is not None:
        return [str(sample_id)]
    return ["unknown"]


def current_video_id():
    return getattr(_STATE, "video_id", None)


def current_meta() -> dict:
    meta = {
        "videoID": getattr(_STATE, "video_id", None),
        "question_id": getattr(_STATE, "question_id", None),
        "sample_id": getattr(_STATE, "sample_id", None),
    }
    extra = getattr(_STATE, "extra", None) or {}
    meta.update(extra)
    return {k: v for k, v in meta.items() if v is not None}
