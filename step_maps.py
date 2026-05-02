"""Progress step maps aligned with extract_tenses.py log lines."""

from tense_constants import ALL_TENSES


def _tense_labels() -> dict[str, str]:
    return {
        "present_simple": "Present Simple",
        "present_continuous": "Present Continuous",
        "past_simple": "Past Simple",
        "past_continuous": "Past Continuous",
        "present_perfect": "Present Perfect",
        "present_perfect_continuous": "Present Perfect Continuous",
        "past_perfect": "Past Perfect",
        "future_simple": "Future Simple",
        "future_continuous": "Future Continuous",
        "future_perfect": "Future Perfect",
        "future_perfect_continuous": "Future Perfect Continuous",
        "future_going_to": "Future (going to)",
    }


def build_extraction_step_map() -> dict[str, tuple[int, str]]:
    """Map log substrings to (progress_pct, status_text) for full extract (URL or local)."""
    labels = _tense_labels()
    m: dict[str, tuple[int, str]] = {
        "[yt-dlp]": (10, "Downloading video…"),
        "[local]": (12, "Using local file…"),
        "[download]  100%": (40, "Download complete"),
        "[Whisper] Reusing": (68, "Reusing saved transcript…"),
        "[Whisper] Transcribing": (50, "Transcribing with Whisper (this may take a while)…"),
        "[Whisper] Done": (70, "Transcription done"),
    }
    n = len(ALL_TENSES)
    base, span = 72, 27
    for i, tense in enumerate(ALL_TENSES):
        pct = base + max(1, int((i + 1) * span / n))
        human = labels.get(tense, tense)
        m[f"[{tense}]"] = (pct, f"Extracting {human} clips…")
    m["✅ Done!"] = (100, "All clips extracted!")
    return m


def build_reanalyze_step_map() -> dict[str, tuple[int, str]]:
    """Progress map when re-running on an existing file (no yt-dlp)."""
    labels = _tense_labels()
    m: dict[str, tuple[int, str]] = {
        "[Whisper] Reusing": (15, "Reusing saved transcript…"),
        "[Whisper] Transcribing": (35, "Transcribing…"),
        "[Whisper] Done": (55, "Transcript ready"),
    }
    n = len(ALL_TENSES)
    base, span = 58, 40
    for i, tense in enumerate(ALL_TENSES):
        pct = base + max(1, int((i + 1) * span / n))
        human = labels.get(tense, tense)
        m[f"[{tense}]"] = (pct, f"Extracting {human} clips…")
    m["✅ Done!"] = (100, "Re-analysis complete!")
    return m
