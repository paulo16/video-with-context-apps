"""
Extract video clips where specific English tenses are used.

Detects:
  - present_simple          She works every day.
  - present_continuous      She is working now.
  - past_simple             She worked yesterday.
  - past_continuous         She was working when I called.
  - present_perfect         She has worked here for years.
  - present_perfect_cont    She has been working all day.
  - past_perfect            She had worked before I arrived.
  - future_going_to         She is going to work tomorrow.

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
import tempfile

# Force UTF-8 output so filenames with non-ASCII chars don't crash on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
import whisper
import spacy

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_CLIP_DURATION = 30  # seconds — short enough to stay focused
DEFAULT_MAX_CLIPS_PER_TENSE = 5  # cap to avoid generating hundreds of clips
WHISPER_MODEL = "base"
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


check_ffmpeg()


ALL_TENSES = [
    "present_simple",
    "present_continuous",
    "past_simple",
    "past_continuous",
    "present_perfect",
    "present_perfect_continuous",
    "past_perfect",
    "future_going_to",
]

nlp = spacy.load("en_core_web_sm")


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

    # Build yt-dlp command with all anti-blocking options
    cmd = [
        "yt-dlp",
        "--format",
        "18",  # Use format 18 (MP4 360p) which is most compatible
        "--output",
        "%(title)s.%(ext)s",
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

    # Fallback: find the most recently modified mp4 in cwd
    mp4_files = sorted(
        [f for f in os.listdir(".") if f.endswith(".mp4")],
        key=lambda f: os.path.getmtime(f),
        reverse=True,
    )
    if not mp4_files:
        raise FileNotFoundError(
            "yt-dlp finished but no .mp4 file found in current directory."
        )
    path = mp4_files[0]
    print(f"[yt-dlp] Saved: {path}")
    return path


# ── Transcribe with Whisper ───────────────────────────────────────────────────


def transcribe(video_path: str) -> list[dict]:
    """Transcribe video; reuse saved transcript if available."""
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{stem}.json")

    if os.path.exists(transcript_path):
        print(f"[Whisper] Reusing saved transcript: {transcript_path}")
        with open(transcript_path, encoding="utf-8") as f:
            return json.load(f)

    print(f"[Whisper] Transcribing (model={WHISPER_MODEL}) …")
    model = whisper.load_model(WHISPER_MODEL)
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
):
    """Cut [start, end] from source and save to out_path.
    If context is provided, subtitles are burned in (re-encode with libx264).
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
            subprocess.run(
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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)
    else:
        subprocess.run(
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# ── Main ─────────────────────────────────────────────────────────────────────


def process(
    source: str,
    clip_duration: int = DEFAULT_CLIP_DURATION,
    burn_subtitles: bool = True,
    max_clips_per_tense: int = DEFAULT_MAX_CLIPS_PER_TENSE,
):
    half = clip_duration // 2

    if source.startswith("http"):
        video_path = download_video(source)
    else:
        video_path = source

    segments = transcribe(video_path)

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
            extract_clip(
                video_path,
                clip_start,
                clip_end,
                out_path,
                context=context if burn_subtitles else None,
            )

    # Merge: keep clips from other videos, replace this video's clips
    kept = [c for c in existing if c.get("source_video") != video_stem]
    merged = kept + hits

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! {len(hits)} clips (duration={clip_duration}s) → {OUTPUT_DIR}/")
    for t in ALL_TENSES:
        if counters[t]:
            print(f"   {t:<36} {counters[t]}")
    print(f"   Summary: {summary_path}")


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
    args = parser.parse_args()
    process(
        args.source,
        clip_duration=args.clip_duration,
        burn_subtitles=not args.no_subtitles,
        max_clips_per_tense=args.max_clips_per_tense,
    )
