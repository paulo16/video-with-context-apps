"""Shared constants for tense extraction and Streamlit UI (no heavy imports)."""

import os

# Video library layout under project root
VIDEOS_ROOT = "videos"
VIDEOS_DOWNLOADS_DIR = os.path.join(VIDEOS_ROOT, "downloads")
VIDEOS_UPLOADS_DIR = os.path.join(VIDEOS_ROOT, "uploads")

ALL_TENSES = [
    "present_simple",
    "present_continuous",
    "past_simple",
    "past_continuous",
    "present_perfect",
    "present_perfect_continuous",
    "past_perfect",
    "future_simple",
    "future_continuous",
    "future_perfect",
    "future_perfect_continuous",
    "future_going_to",
]

DEFAULT_WHISPER_MODEL = "base"
VALID_WHISPER_MODELS = ("tiny", "base", "small", "medium", "large", "turbo")


def normalize_whisper_model(name: str) -> str:
    n = (name or DEFAULT_WHISPER_MODEL).strip().lower()
    if n not in VALID_WHISPER_MODELS:
        return DEFAULT_WHISPER_MODEL
    return n


def list_source_mp4_rel_paths(videos_root: str = VIDEOS_ROOT) -> list[str]:
    """Paths relative to videos_root: uploads/…, downloads/…, or legacy root *.mp4."""
    if not os.path.isdir(videos_root):
        return []
    rel: list[tuple[str, float]] = []
    for name in os.listdir(videos_root):
        path = os.path.join(videos_root, name)
        if os.path.isfile(path) and name.lower().endswith(".mp4"):
            rel.append((name, os.path.getmtime(path)))
    for sub in ("uploads", "downloads"):
        subdir = os.path.join(videos_root, sub)
        if not os.path.isdir(subdir):
            continue
        for name in os.listdir(subdir):
            if not name.lower().endswith(".mp4"):
                continue
            p = os.path.join(subdir, name)
            if os.path.isfile(p):
                r = f"{sub}/{name}".replace("\\", "/")
                rel.append((r, os.path.getmtime(p)))
    rel.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in rel]
