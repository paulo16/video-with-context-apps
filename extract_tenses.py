"""
Extract video clips where specific English tenses are used.

Detects:
  - present_simple          She works every day.
  - present_continuous      She is working now.
  - past_simple             She worked yesterday.
  - past_continuous         She was working when I called.
  - present_perfect         She has worked here for years.
  - present_perfect_cont    She has been working all day.
  - past_perfect            She had worked before I arrived.  - future_simple           She will work tomorrow.
  - future_continuous       She will be working at 5 PM.
  - future_perfect          She will have finished by then.
  - future_perfect_cont     She will have been working for hours.  - future_going_to         She is going to work tomorrow.

Clip duration: default 30 s (configurable with --clip-duration).
Full transcript is saved to transcripts/<video>.json and reused on re-runs.

Usage:
    python extract_tenses.py <youtube_url_or_local_file> [--clip-duration N]
"""

import sys
import os
import json
import subprocess
import re
# Force UTF-8 output so filenames with non-ASCII chars don't crash on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
import whisper
import spacy

from tense_constants import (
    ALL_TENSES,
    DEFAULT_WHISPER_MODEL,
    VALID_WHISPER_MODELS,
    VIDEOS_DOWNLOADS_DIR,
    VIDEOS_ROOT,
    normalize_whisper_model,
)

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_CLIP_DURATION = 30  # seconds — short enough to stay focused
DEFAULT_MAX_CLIPS_PER_TENSE = 5  # cap to avoid generating hundreds of clips
OUTPUT_DIR = "clips"
TRANSCRIPTS_DIR = "transcripts"  # saved per-video transcripts (reused on re-runs)
# ──────────────────────────────────────────────────────────────────────────────


# ── Check system dependencies ────────────────────────────────────────────────
def check_ffmpeg():
    """Verify ffmpeg is available (required by Whisper for audio processing)."""
    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "❌ ffmpeg is not installed or not in PATH.\n"
            "ffmpeg is required to transcribe audio from video files.\n"
            "On Streamlit Cloud, this should be installed via setup.sh.\n"
            "Please contact support or try uploading a different file."
        )


nlp = spacy.load("en_core_web_sm")

MIN_CLIP_FILE_BYTES = 256


# ── Tense detection ──────────────────────────────────────────────────────────


def classify_tense(sent_text: str) -> list[str]:
    """Return list of tenses found in a sentence (no duplicates)."""
    found = set()
    doc = nlp(sent_text)
    tokens = list(doc)
    n = len(tokens)

    def tag(i):
        return tokens[i].tag_ if i < n else ""

    def lem(i):
        return tokens[i].lemma_.lower() if i < n else ""

    def txt(i):
        return tokens[i].text.lower() if i < n else ""

    for i, tok in enumerate(tokens):
        t = tok.tag_
        l = tok.lemma_.lower()

        # ── future going to: (am/is/are) + going + to + VB ─────────────
        if t in ("VBZ", "VBP", "VBD") and l == "be":
            for j in range(i + 1, min(i + 4, n)):
                if txt(j) == "going":
                    if j + 1 < n and txt(j + 1) == "to":
                        if j + 2 < n and tag(j + 2) in ("VB", "VBZ", "VBP"):
                            found.add("future_going_to")

        # ── future forms with will/shall ─────────────────────────────
        if t == "MD" and txt(i) in ("will", "shall"):
            # future perfect continuous: will have been VBG
            for j in range(i + 1, min(i + 6, n)):
                if txt(j) == "have":
                    for k in range(j + 1, min(j + 6, n)):
                        if txt(k) == "been":
                            for m in range(k + 1, min(k + 5, n)):
                                if tag(m) == "VBG":
                                    found.add("future_perfect_continuous")
                                    break
                            break
                    if "future_perfect_continuous" in found:
                        break
            # future perfect: will have VBN (not been)
            if "future_perfect_continuous" not in found:
                for j in range(i + 1, min(i + 5, n)):
                    if txt(j) == "have":
                        for k in range(j + 1, min(j + 5, n)):
                            if tag(k) == "VBN" and txt(k) != "been":
                                found.add("future_perfect")
                                break
                        break
            # future continuous: will be VBG
            for j in range(i + 1, min(i + 5, n)):
                if txt(j) == "be":
                    for k in range(j + 1, min(j + 5, n)):
                        if tag(k) == "VBG":
                            found.add("future_continuous")
                            break
                    break
            # future simple: will + VB (if not captured by other future forms)
            if not any(
                tense in found
                for tense in (
                    "future_perfect_continuous",
                    "future_perfect",
                    "future_continuous",
                )
            ):
                for j in range(i + 1, min(i + 4, n)):
                    if tag(j) == "VB":
                        found.add("future_simple")
                        break

        # ── present perfect continuous: have/has + been + VBG ──────────
        if t in ("VBZ", "VBP") and l == "have":
            for j in range(i + 1, min(i + 5, n)):
                if txt(j) == "been":
                    for k in range(j + 1, min(j + 4, n)):
                        if tag(k) == "VBG":
                            found.add("present_perfect_continuous")
                    break

        # ── present perfect: have/has + VBN (not been) ───────────────
        if t in ("VBZ", "VBP") and l == "have":
            if "present_perfect_continuous" not in found:
                for j in range(i + 1, min(i + 5, n)):
                    if tag(j) == "VBN" and txt(j) != "been":
                        found.add("present_perfect")
                        break

        # ── past perfect: had + VBN ──────────────────────────────
        if t == "VBD" and l == "have":
            for j in range(i + 1, min(i + 5, n)):
                if tag(j) == "VBN":
                    found.add("past_perfect")
                    break

        # ── present continuous: am/is/are + VBG ────────────────────
        if t in ("VBZ", "VBP") and l == "be":
            for j in range(i + 1, min(i + 4, n)):
                if tag(j) == "VBG":
                    found.add("present_continuous")
                    break

        # ── past continuous: was/were + VBG ──────────────────────
        if t == "VBD" and l == "be":
            for j in range(i + 1, min(i + 4, n)):
                if tag(j) == "VBG":
                    found.add("past_continuous")
                    break

        # ── past simple: VBD main verb (not be/have) ─────────────────
        if (
            t == "VBD"
            and l not in ("be", "have")
            and tok.dep_ in ("ROOT", "relcl", "advcl", "ccomp", "xcomp")
        ):
            found.add("past_simple")

        # ── present simple: VBZ/VBP main verb (not be/have) ───────────
        if (
            t in ("VBZ", "VBP")
            and l not in ("be", "have")
            and tok.dep_ in ("ROOT", "relcl", "advcl", "ccomp", "xcomp")
        ):
            found.add("present_simple")

    return list(found)


# ── Download with yt-dlp ─────────────────────────────────────────────────────


def download_video(source: str) -> str:
    """Download video from URL or return local file path."""
    # Check if it's a local file
    if os.path.isfile(source):
        print(f"[local] Using local file: {source}")
        return source

    # Otherwise, treat as URL and download
    print(f"[yt-dlp] Downloading: {source}")

    os.makedirs(VIDEOS_DOWNLOADS_DIR, exist_ok=True)
    output_template = os.path.join(VIDEOS_DOWNLOADS_DIR, "%(title)s.%(ext)s")

    # Build yt-dlp command with all anti-blocking options
    cmd = [
        "yt-dlp",
        "--format",
        "18",  # Use format 18 (MP4 360p) which is most compatible
        "--output",
        output_template,
        "--user-agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "--extractor-args",
        "youtube:player_client=android_music",
        "--socket-timeout",
        "60",
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--retry-sleep",
        "5",
        "--force-ipv4",
        "--skip-unavailable-fragments",
        source,
    ]

    # Try to get filename (non-critical, so catch errors)
    probe = subprocess.run(
        cmd[:-1] + ["--print", "filename", "--no-download"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    expected_path = (
        probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else None
    )

    # Actually download
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except subprocess.CalledProcessError as e:
        if "403" in str(e) or "Forbidden" in str(e):
            raise RuntimeError(
                "YouTube blocked the download (HTTP 403). "
                "This is a temporary YouTube restriction. "
                "Please try again later or use local file upload instead."
            )
        raise

    # Confirm the file exists
    if expected_path and os.path.isfile(expected_path):
        print(f"[yt-dlp] Saved: {expected_path}")
        return expected_path

    # Fallback: newest mp4 in downloads/, then legacy flat videos/ root
    candidates = []
    if os.path.isdir(VIDEOS_DOWNLOADS_DIR):
        candidates.extend(
            os.path.join(VIDEOS_DOWNLOADS_DIR, f)
            for f in os.listdir(VIDEOS_DOWNLOADS_DIR)
            if f.endswith(".mp4")
        )
    if os.path.isdir(VIDEOS_ROOT):
        candidates.extend(
            os.path.join(VIDEOS_ROOT, f)
            for f in os.listdir(VIDEOS_ROOT)
            if f.endswith(".mp4") and os.path.isfile(os.path.join(VIDEOS_ROOT, f))
        )
    mp4_files = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    if not mp4_files:
        raise FileNotFoundError(
            "yt-dlp finished but no .mp4 file found under videos/downloads or videos/."
        )
    path = mp4_files[0]
    print(f"[yt-dlp] Saved: {path}")
    return path


# ── Transcribe with Whisper ───────────────────────────────────────────────────


def transcribe(video_path: str, whisper_model: str = DEFAULT_WHISPER_MODEL) -> list[dict]:
    """Transcribe video; reuse saved transcript if available."""
    wm = normalize_whisper_model(whisper_model)
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{stem}.json")

    if os.path.exists(transcript_path):
        print(f"[Whisper] Reusing saved transcript: {transcript_path}")
        with open(transcript_path, encoding="utf-8") as f:
            return json.load(f)

    print(f"[Whisper] Transcribing (model={wm}) …")
    model = whisper.load_model(wm)
    result = model.transcribe(video_path, language="en", word_timestamps=False)
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in result["segments"]
    ]
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"[Whisper] Done — {len(segments)} segments → {transcript_path}")
    return segments


# ── Extract clips with FFmpeg ─────────────────────────────────────────────────


def _secs_to_ass(secs: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cc"""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _build_ass(context: list, clip_start: float) -> str:
    """Generate ASS subtitle content.
    Normal lines → white.  Highlighted line → bold yellow.
    """
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1280\n"
        "PlayResY: 720\n"
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Normal,Arial,34,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,30,30,40,1\n"
        "Style: Highlight,Arial,38,&H0000FFFF,&H000000FF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,2,1,2,30,30,40,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for seg in context:
        t0 = max(0.0, seg["start"] - clip_start)
        t1 = max(t0 + 0.1, seg["end"] - clip_start)
        style = "Highlight" if seg.get("highlight") else "Normal"
        text = seg["text"].replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        lines.append(
            f"Dialogue: 0,{_secs_to_ass(t0)},{_secs_to_ass(t1)},{style},,0,0,0,,{text}"
        )
    return header + "\n".join(lines) + "\n"


def extract_clip(
    source: str,
    start: float,
    end: float,
    out_path: str,
    context: list | None = None,
) -> bool:
    """Cut [start, end] from source and save to out_path.
    If context is provided, subtitles are burned in (re-encode with libx264).
    Returns True if the output file exists and is non-trivial size.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if context:
        # Write ASS to a relative path (no drive letter colon) so FFmpeg's
        # ass= filter can parse the path without escaping issues on Windows.
        ass_path = os.path.join(os.path.dirname(out_path) or ".", "_tmp_sub.ass")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(_build_ass(context, start))
        # Use forward slashes for the filter path (FFmpeg on Windows accepts these)
        ass_filter_path = ass_path.replace("\\", "/")
        try:
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(max(0, start)),
                    "-to",
                    str(end),
                    "-i",
                    source,
                    "-vf",
                    f"ass={ass_filter_path}",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-preset",
                    "fast",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    out_path,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if r.returncode != 0:
                err = (r.stderr or "").strip()[-4000:]
                print(f"[ffmpeg] exit={r.returncode} stderr (tail):\n{err}", flush=True)
                return False
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)
    else:
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(max(0, start)),
                "-to",
                str(end),
                "-i",
                source,
                "-c",
                "copy",
                out_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if r.returncode != 0:
            err = (r.stderr or "").strip()[-4000:]
            print(f"[ffmpeg] exit={r.returncode} stderr (tail):\n{err}", flush=True)
            return False

    if not os.path.isfile(out_path) or os.path.getsize(out_path) < MIN_CLIP_FILE_BYTES:
        print(
            f"[ffmpeg] Output missing or too small (<{MIN_CLIP_FILE_BYTES} B): {out_path}",
            flush=True,
        )
        return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────


def merge_clips_for_source(existing: list, hits: list, video_stem: str) -> list:
    """Keep clips from other sources; replace entries for video_stem with hits."""
    kept = [c for c in existing if c.get("source_video") != video_stem]
    return kept + hits


def process(
    source: str,
    clip_duration: int = DEFAULT_CLIP_DURATION,
    burn_subtitles: bool = True,
    max_clips_per_tense: int = DEFAULT_MAX_CLIPS_PER_TENSE,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
):
    half = clip_duration // 2

    if source.startswith("http"):
        video_path = download_video(source)
    else:
        video_path = source

    segments = transcribe(video_path, whisper_model=whisper_model)

    counters = {t: 0 for t in ALL_TENSES}
    hits = []

    # Load existing summary to append (don't overwrite clips from other videos)
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    existing = []
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            existing = json.load(f)

    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        tenses = classify_tense(text)
        for tense in tenses:
            if counters[tense] >= max_clips_per_tense:
                continue
            mid = (seg["start"] + seg["end"]) / 2
            clip_start = max(0, mid - half)
            clip_end = clip_start + clip_duration
            counters[tense] += 1
            out_name = f"{tense}/{video_stem}_clip_{counters[tense]:03d}.mp4"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            context = [
                {
                    "start": round(s["start"], 2),
                    "end": round(s["end"], 2),
                    "text": s["text"].strip(),
                    "highlight": s["text"].strip() == text,
                }
                for s in segments
                if s["start"] >= clip_start
                and s["end"] <= clip_end
                and s["text"].strip()
            ]

            hits.append(
                {
                    "tense": tense,
                    "sentence": text,
                    "time": f"{seg['start']:.1f}s",
                    "clip_start": clip_start,
                    "clip": out_path,
                    "context": context,
                    "source_video": video_stem,
                }
            )
            print(f"[{tense}] {text[:70]}  → {out_path}")
            ok = extract_clip(
                video_path,
                clip_start,
                clip_end,
                out_path,
                context=context if burn_subtitles else None,
            )
            if not ok:
                print(f"[ERROR] Failed to write clip file: {out_path}", flush=True)

    merged = merge_clips_for_source(existing, hits, video_stem)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    abs_clips = os.path.abspath(OUTPUT_DIR)
    abs_source = os.path.abspath(video_path)
    abs_summary = os.path.abspath(summary_path)
    print(f"\n✅ Done! {len(hits)} clips (duration={clip_duration}s) → {OUTPUT_DIR}/")
    for t in ALL_TENSES:
        if counters[t]:
            print(f"   {t:<36} {counters[t]}")
    print(f"   Summary: {summary_path}")
    print(f"[saved] Full source video (kept on disk): {abs_source}")
    print(
        f"[saved] Cut clips (MP4, burned subs if enabled): {abs_clips}/<tense>/*.mp4"
    )
    print(f"[saved] Browse index (metadata): {abs_summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract English tense clips")
    parser.add_argument("source", help="YouTube URL or local video file")
    parser.add_argument(
        "--clip-duration",
        type=int,
        default=DEFAULT_CLIP_DURATION,
        help=f"Clip length in seconds (default {DEFAULT_CLIP_DURATION})",
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Skip burning subtitles into clips (faster, uses stream copy)",
    )
    parser.add_argument(
        "--max-clips-per-tense",
        type=int,
        default=DEFAULT_MAX_CLIPS_PER_TENSE,
        help=f"Max clips extracted per tense (default {DEFAULT_MAX_CLIPS_PER_TENSE})",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model name (default {DEFAULT_WHISPER_MODEL}). Options: "
        + ", ".join(VALID_WHISPER_MODELS),
    )
    args = parser.parse_args()
    check_ffmpeg()
    process(
        args.source,
        clip_duration=args.clip_duration,
        burn_subtitles=not args.no_subtitles,
        max_clips_per_tense=args.max_clips_per_tense,
        whisper_model=normalize_whisper_model(args.whisper_model),
    )
