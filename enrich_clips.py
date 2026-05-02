"""enrich_clips.py — Transcribe existing 1-min clips and update summary.json

Clips extracted before transcript support was added have no 'context' field.
This script runs Whisper on each such clip file (≈1 min each, fast) and
updates summary.json in-place, one clip at a time.

Usage:  python enrich_clips.py
"""

import argparse
import json
import os
import sys

import whisper

from tense_constants import DEFAULT_WHISPER_MODEL, normalize_whisper_model

SUMMARY_FILE = "clips/summary.json"


def sentence_matches(detected: str, seg_text: str) -> bool:
    """True if the Whisper segment corresponds to the detected sentence."""
    a = detected.strip().lower()
    b = seg_text.strip().lower()
    return a == b or a in b or b in a


def main():
    parser = argparse.ArgumentParser(description="Enrich clips with Whisper transcripts")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Only enrich clip at this zero-based index in summary.json",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model (default {DEFAULT_WHISPER_MODEL})",
    )
    args, _unknown = parser.parse_known_args()
    whisper_model = normalize_whisper_model(args.whisper_model)

    if not os.path.exists(SUMMARY_FILE):
        print("No summary.json found. Run an extraction first.")
        sys.exit(1)

    with open(SUMMARY_FILE, encoding="utf-8") as f:
        clips = json.load(f)

    target_index = args.index

    if target_index is not None:
        if clips[target_index].get("context"):
            print(f"Clip {target_index} already has a transcript.")
            sys.exit(0)
        missing = [target_index]
    else:
        missing = [i for i, c in enumerate(clips) if not c.get("context")]
        if not missing:
            print("All clips already have transcripts. Nothing to do.")
            sys.exit(0)

    print(f"[Whisper] Loading model '{whisper_model}'…")
    model = whisper.load_model(whisper_model)
    print(f"[Whisper] Model ready. {len(missing)} clips to enrich.\n")

    for done, idx in enumerate(missing, 1):
        clip = clips[idx]
        video_path = clip["clip"].replace("\\", "/")

        if not os.path.exists(video_path):
            print(
                f"[{done}/{len(missing)}] SKIP (file missing): {video_path}", flush=True
            )
            continue

        print(
            f"[{done}/{len(missing)}] Transcribing: {os.path.basename(video_path)}",
            flush=True,
        )
        result = model.transcribe(video_path, language="en", word_timestamps=False)

        detected = clip["sentence"].strip()
        context = []
        for seg in result["segments"]:
            text = seg["text"].strip()
            if not text:
                continue
            context.append(
                {
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": text,
                    "highlight": sentence_matches(detected, text),
                }
            )

        # If nothing matched highlight, mark the segment closest to clip mid
        if not any(s["highlight"] for s in context) and context:
            mid = result["segments"][-1]["end"] / 2
            closest = min(context, key=lambda s: abs((s["start"] + s["end"]) / 2 - mid))
            closest["highlight"] = True

        clip["context"] = context
        clip.setdefault("clip_start", 0)  # clip file timestamps start at 0

        # Save after every clip so progress survives interruption
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(clips, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! {len(missing)} clips enriched → {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
