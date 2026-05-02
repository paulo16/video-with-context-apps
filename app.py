import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import streamlit as st

from step_maps import build_extraction_step_map, build_reanalyze_step_map
from streamlit_subprocess import drain_subprocess_queue, start_stream_reader
from tense_constants import (
    DEFAULT_WHISPER_MODEL,
    VALID_WHISPER_MODELS,
    VIDEOS_DOWNLOADS_DIR,
    VIDEOS_ROOT,
    VIDEOS_UPLOADS_DIR,
    list_source_mp4_rel_paths,
)

# ── Config ────────────────────────────────────────────────────────────────────
SUMMARY_FILE = "clips/summary.json"
PYTHON = sys.executable
TENSE_LABELS = {
    "present_simple": "🟡 Present Simple",
    "present_continuous": "🟠 Present Continuous",
    "past_simple": "🔵 Past Simple",
    "past_continuous": "🟣 Past Continuous",
    "present_perfect": "🟢 Present Perfect",
    "present_perfect_continuous": "🟠 Present Perfect Continuous",
    "past_perfect": "🟤 Past Perfect",
    "future_simple": "⬜ Future Simple",
    "future_continuous": "🟠 Future Continuous",
    "future_perfect": "🟢 Future Perfect",
    "future_perfect_continuous": "🟠 Future Perfect Continuous",
    "future_going_to": "⬛ Future (going to)",
}
TENSE_COLORS = {
    "present_simple": "#5f5f1e",
    "present_continuous": "#5f3a1e",
    "past_simple": "#1e3a5f",
    "past_continuous": "#3a1e5f",
    "present_perfect": "#1e5f3a",
    "present_perfect_continuous": "#5f4a1e",
    "past_perfect": "#4a1e5f",
    "future_simple": "#555555",
    "future_continuous": "#774411",
    "future_perfect": "#117744",
    "future_perfect_continuous": "#774477",
    "future_going_to": "#2a2a2a",
}
TENSE_EXPLANATIONS = {
    "present_simple": "Habitual / general truth. → *She **works** every day.*",
    "present_continuous": "Action happening now. → *She **is working** right now.*",
    "past_simple": "Completed past action. → *She **walked** into the room.*",
    "past_continuous": "Ongoing action in the past. → *She **was working** when I called.*",
    "present_perfect": "Past action with present relevance. → *I **have seen** that movie.*",
    "present_perfect_continuous": "Started in the past, still ongoing. → *She **has been waiting** for hours.*",
    "past_perfect": "Action before another past action. → *She **had worked** before I arrived.*",
    "future_simple": "Future action expressed with will/shall. → *She **will work** tomorrow.*",
    "future_continuous": "Ongoing future action. → *She **will be working** at 5 PM.*",
    "future_perfect": "Action completed before a future moment. → *She **will have finished** by then.*",
    "future_perfect_continuous": "Action ongoing until a future moment. → *She **will have been working** for hours.*",
    "future_going_to": "Planned future. → *She **is going to work** tomorrow.*",
}

_APP_DIR = os.path.dirname(os.path.abspath(__file__)) or "."


# ── Load / save data ─────────────────────────────────────────────────────────
@st.cache_data
def load_clips():
    if not os.path.exists(SUMMARY_FILE):
        return []
    with open(SUMMARY_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_clips(clips_list):
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(clips_list, f, ensure_ascii=False, indent=2)
    load_clips.clear()


def _stream_process(proc, q):
    """Read process stdout+stderr line by line and push to queue."""
    for line in iter(proc.stdout.readline, ""):
        q.put(("line", line.rstrip()))
    proc.wait()
    q.put(("exit", proc.returncode))


def _whisper_model_arg():
    return str(st.session_state.get("whisper_model", DEFAULT_WHISPER_MODEL))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="English Tenses Explorer",
    page_icon="🎬",
    layout="wide",
)

os.makedirs(VIDEOS_ROOT, exist_ok=True)
os.makedirs(VIDEOS_UPLOADS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DOWNLOADS_DIR, exist_ok=True)

st.sidebar.header("Pipeline")
st.sidebar.selectbox(
    "Whisper model",
    options=list(VALID_WHISPER_MODELS),
    index=list(VALID_WHISPER_MODELS).index(DEFAULT_WHISPER_MODEL),
    key="whisper_model",
    help="Smaller models run faster; larger models are more accurate.",
)

st.title("🎬 English Tenses Explorer")
st.caption(
    "Clips automatically extracted from real conversations — organized by grammatical tense."
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_extract, tab_reanalyze, tab_browse = st.tabs(
    ["📥 Extract New Video", "🔄 Re-analyze Existing Video", "🎬 Browse Clips"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXTRACT NEW VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_extract:
    st.subheader("Extract tenses from a video")

    with st.expander(
        "⚠️ **YouTube download issues?** (click to expand)", expanded=False
    ):
        st.warning(
            "YouTube sometimes blocks automated downloads on cloud servers. "
            "**We recommend uploading a local video file instead** for the most reliable experience. "
            "If you must use YouTube, try again later or use a different URL."
        )

    input_method = st.radio(
        "Choose video source:",
        ["Upload local file", "YouTube URL"],
        horizontal=True,
        key="input_method",
    )

    if input_method == "Upload local file":
        st.markdown(
            "Upload a local video file (MP4, AVI, MOV, etc.). The script will **transcribe → detect tenses → cut clips** automatically."
        )
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key="video_upload",
        )
        if uploaded_file is not None:
            orig = Path(uploaded_file.name).name
            ext = Path(orig).suffix.lower()
            if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                ext = ".mp4"
            safe_name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:10]}{ext}"
            temp_path = os.path.join(VIDEOS_UPLOADS_DIR, safe_name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source_path = temp_path
            st.success(
                f"✅ File uploaded: **{orig}** (saved as `videos/uploads/{safe_name}`)"
            )
        else:
            source_path = ""
    else:
        st.markdown(
            "⚠️ Paste a YouTube URL. Note: This may fail if YouTube blocks the request. "
            "If you get a 403 error, please use local file upload instead."
        )
        url_input = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="yt_url",
        )
        source_path = url_input

    clip_duration = st.slider(
        "⏱ Clip duration (seconds)",
        min_value=15,
        max_value=90,
        value=30,
        step=5,
        help="How long each extracted clip will be. 30s is ideal to focus on one sentence.",
        key="clip_dur",
    )
    burn_subs = st.toggle(
        "📝 Burn subtitles into video",
        value=True,
        help="Subtitles are hardcoded into the clip. The detected sentence appears in yellow. Slightly slower (~5s/clip).",
        key="burn_subs",
    )
    max_clips = st.slider(
        "📄 Max clips per tense",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Caps how many clips are generated per tense. Lower this if a run takes too long.",
        key="max_clips",
    )

    col_btn, _ = st.columns([1, 5])
    start_btn = col_btn.button(
        "▶ Start extraction", type="primary", disabled=not source_path.strip()
    )

    if "running" not in st.session_state:
        st.session_state.running = False
    if "log_lines" not in st.session_state:
        st.session_state.log_lines = []
    if "done" not in st.session_state:
        st.session_state.done = False

    if start_btn and source_path.strip():
        st.session_state.log_lines = []
        st.session_state.running = True
        st.session_state.done = False

        wm = _whisper_model_arg()
        proc = subprocess.Popen(
            [
                PYTHON,
                "extract_tenses.py",
                source_path.strip(),
                "--clip-duration",
                str(clip_duration),
                "--max-clips-per-tense",
                str(max_clips),
                "--whisper-model",
                wm,
            ]
            + (["--no-subtitles"] if not burn_subs else []),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=_APP_DIR,
        )

        q: queue.Queue = queue.Queue()
        start_stream_reader(proc, q)

        st.markdown("#### Live log")
        log_box = st.empty()
        progress_bar = st.progress(0, text="Starting…")
        step_map = build_extraction_step_map()
        progress_holder = [0]

        def _render_log(text: str):
            log_box.code(text, language="bash")

        rc = drain_subprocess_queue(
            q,
            log_lines=st.session_state.log_lines,
            render_log=_render_log,
            progress_bar=progress_bar,
            step_map=step_map,
            progress_holder=progress_holder,
        )

        if rc == 0:
            progress_bar.progress(100, text="✅ Extraction complete!")
            st.success(
                "Done! Switch to the **Browse Clips** tab to see the results."
            )
            load_clips.clear()
        else:
            progress_bar.progress(
                progress_holder[0], text="❌ Error during extraction"
            )
            full_log = "\n".join(st.session_state.log_lines)
            if (
                "ffmpeg" in full_log.lower()
                or "No such file or directory: 'ffmpeg'" in full_log
            ):
                st.error(
                    "❌ **ffmpeg is not installed**\n\n"
                    "This is a server configuration issue. ffmpeg is required to extract audio from videos.\n\n"
                    "**What you can do:**\n"
                    "1. Try again in a few minutes (the server may be updating)\n"
                    "2. Contact Streamlit Cloud support\n"
                    "3. Try uploading a different video file\n\n"
                    "Check the log below for more details."
                )
            elif "403" in full_log or "Forbidden" in full_log:
                st.error(
                    "❌ **YouTube blocked the download (HTTP 403)**\n\n"
                    "This is a temporary YouTube protection. Try one of these:\n"
                    "1. **Wait a few minutes and try again**\n"
                    "2. **Use a different YouTube video**\n"
                    "3. **Upload a local video file instead** (most reliable)\n\n"
                    "Check the log below for more details."
                )
            else:
                st.error(
                    "❌ Something went wrong. Check the log above for details."
                )
        st.session_state.running = False
        st.session_state.done = True

    elif st.session_state.done and st.session_state.log_lines:
        st.markdown("#### Last extraction log")
        st.code("\n".join(st.session_state.log_lines[-80:]), language="bash")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RE-ANALYZE EXISTING VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_reanalyze:
    st.subheader("🔄 Re-analyze an already-downloaded video")
    st.markdown(
        "If you already have a `.mp4` file locally, you can **re-run the tense analysis** "
        "without downloading again. The transcript is reused if already saved (fast). "
        "YouTube downloads live under `videos/downloads/`; uploads under `videos/uploads/`."
    )

    local_mp4s = list_source_mp4_rel_paths()

    if not local_mp4s:
        st.info(
            "No `.mp4` files found under `videos/` (uploads, downloads, or legacy root). "
            "Download a video or upload one first."
        )
    else:
        selected_mp4 = st.selectbox(
            "Select a video to re-analyze",
            options=local_mp4s,
            key="reanalyze_mp4",
        )

        reanalyze_clip_dur = st.slider(
            "⏱ Clip duration (seconds)",
            min_value=15,
            max_value=90,
            value=30,
            step=5,
            key="reanalyze_clip_dur",
        )
        reanalyze_burn_subs = st.toggle(
            "📝 Burn subtitles into video",
            value=True,
            key="reanalyze_burn_subs",
        )
        reanalyze_max_clips = st.slider(
            "📄 Max clips per tense",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Caps how many clips are generated per tense.",
            key="reanalyze_max_clips",
        )

        ra_btn = st.button("🔄 Re-analyze", type="primary", key="ra_btn")

        if "ra_running" not in st.session_state:
            st.session_state.ra_running = False
        if "ra_log" not in st.session_state:
            st.session_state.ra_log = []
        if "ra_done" not in st.session_state:
            st.session_state.ra_done = False

        if ra_btn and selected_mp4:
            st.session_state.ra_running = True
            st.session_state.ra_done = False
            st.session_state.ra_log = []

            wm = _whisper_model_arg()
            ra_proc = subprocess.Popen(
                [
                    PYTHON,
                    "extract_tenses.py",
                    os.path.join(VIDEOS_ROOT, selected_mp4.replace("/", os.sep)),
                    "--clip-duration",
                    str(reanalyze_clip_dur),
                    "--max-clips-per-tense",
                    str(reanalyze_max_clips),
                    "--whisper-model",
                    wm,
                ]
                + (["--no-subtitles"] if not reanalyze_burn_subs else []),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=_APP_DIR,
            )
            ra_q: queue.Queue = queue.Queue()
            start_stream_reader(ra_proc, ra_q)

            st.markdown("#### Live log")
            ra_log_box = st.empty()
            ra_prog = st.progress(0, text="Starting…")
            step_map_ra = build_reanalyze_step_map()
            ra_progress_holder = [0]

            def _ra_render(text: str):
                ra_log_box.code(text, language="bash")

            ra_rc = drain_subprocess_queue(
                ra_q,
                log_lines=st.session_state.ra_log,
                render_log=_ra_render,
                progress_bar=ra_prog,
                step_map=step_map_ra,
                progress_holder=ra_progress_holder,
            )

            if ra_rc == 0:
                ra_prog.progress(100, text="✅ Re-analysis complete!")
                st.success(
                    "Done! Switch to **Browse Clips** to explore the results."
                )
                load_clips.clear()
            else:
                st.error("Something went wrong. Check the log above.")
            st.session_state.ra_running = False
            st.session_state.ra_done = True

        elif st.session_state.ra_done and st.session_state.ra_log:
            st.markdown("#### Last re-analysis log")
            st.code("\n".join(st.session_state.ra_log[-80:]), language="bash")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BROWSE CLIPS
# ══════════════════════════════════════════════════════════════════════════════
with tab_browse:
    clips = load_clips()

    if not clips:
        st.info("No clips yet. Go to the **Extract New Video** tab to get started.")
        st.stop()

    st.caption(
        "Tense labels come from **spaCy rule-based detection** on each subtitle line — "
        "use them as study hints, not linguistic ground truth."
    )

    missing_count = sum(1 for c in clips if not c.get("context"))
    if missing_count:
        st.info(
            f"**{missing_count} clip(s)** do not have a transcript yet. "
            f"Click below to generate them with Whisper (about 10 s per clip)."
        )
        if "enrich_running" not in st.session_state:
            st.session_state.enrich_running = False
        if "enrich_done" not in st.session_state:
            st.session_state.enrich_done = False
        if "enrich_log" not in st.session_state:
            st.session_state.enrich_log = []

        if not st.session_state.enrich_running:
            if st.button("📝 Generate transcripts", type="primary"):
                st.session_state.enrich_running = True
                st.session_state.enrich_done = False
                st.session_state.enrich_log = []
                enrich_q = queue.Queue()

                wm = _whisper_model_arg()
                enrich_proc = subprocess.Popen(
                    [
                        PYTHON,
                        "enrich_clips.py",
                        "--whisper-model",
                        wm,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    cwd=_APP_DIR,
                )
                enrich_thread = threading.Thread(
                    target=_stream_process, args=(enrich_proc, enrich_q), daemon=True
                )
                enrich_thread.start()

                log_area = st.empty()
                prog = st.progress(0, text="Starting Whisper…")
                total_for_prog = missing_count
                done_for_prog = 0

                while True:
                    msg_type, payload = enrich_q.get()
                    if msg_type == "line":
                        st.session_state.enrich_log.append(payload)
                        log_area.code(
                            "\n".join(st.session_state.enrich_log[-30:]),
                            language="bash",
                        )
                        if payload.startswith("[") and "/" in payload:
                            done_for_prog += 1
                            pct = min(
                                int(done_for_prog / max(total_for_prog, 1) * 100), 99
                            )
                            prog.progress(
                                pct,
                                text=f"Transcribing… {done_for_prog}/{total_for_prog}",
                            )
                    elif msg_type == "exit":
                        if payload == 0:
                            prog.progress(100, text="✅ Transcripts ready!")
                            st.success(
                                "All clips now have transcripts — scroll down to browse."
                            )
                            load_clips.clear()
                        else:
                            st.error("Something went wrong. Check the log above.")
                        st.session_state.enrich_running = False
                        st.session_state.enrich_done = True
                        st.rerun()
                        break
        st.divider()

    st.sidebar.header("Filters")

    all_tenses = sorted(set(c["tense"] for c in clips))
    selected_tenses = st.sidebar.multiselect(
        "Tense",
        options=all_tenses,
        default=all_tenses,
        format_func=lambda t: TENSE_LABELS.get(t, t),
    )

    all_sources = sorted(set(c.get("source_video", "unknown") for c in clips))
    if len(all_sources) > 1:
        selected_sources = st.sidebar.multiselect(
            "📹 Video source",
            options=all_sources,
            default=all_sources,
        )
    else:
        selected_sources = all_sources

    search_query = st.sidebar.text_input(
        "🔍 Search in sentence", placeholder="e.g. went, came, have been..."
    )

    st.sidebar.divider()
    st.sidebar.header("📖 Grammar reference")
    for tense, label in TENSE_LABELS.items():
        with st.sidebar.expander(label):
            st.markdown(TENSE_EXPLANATIONS.get(tense, ""))

    st.sidebar.divider()
    st.sidebar.markdown("### Danger zone")
    if "confirm_delete_all" not in st.session_state:
        st.session_state.confirm_delete_all = False

    if not st.session_state.confirm_delete_all:
        if st.sidebar.button("🗑️ Delete all clips", use_container_width=True):
            st.session_state.confirm_delete_all = True
            st.rerun()
    else:
        st.sidebar.warning(
            f"⚠️ Permanently delete **{len(clips)}** clip(s) and their files?"
        )
        da_c1, da_c2 = st.sidebar.columns(2)
        if da_c1.button("✅ Yes, delete all", type="primary", key="da_yes"):
            for c in clips:
                p = c["clip"]
                if os.path.exists(p):
                    os.remove(p)
            save_clips([])
            st.session_state.confirm_delete_all = False
            st.rerun()
        if da_c2.button("❌ Cancel", key="da_no"):
            st.session_state.confirm_delete_all = False
            st.rerun()

    st.subheader("📊 Summary")
    counts = {t: sum(1 for c in clips if c["tense"] == t) for t in all_tenses}
    stat_cols = st.columns(len(all_tenses) + 1)
    stat_cols[0].metric("Total clips", len(clips))
    for i, tense in enumerate(all_tenses):
        label = TENSE_LABELS.get(tense, tense)
        stat_cols[i + 1].metric(label, counts.get(tense, 0))

    st.divider()

    filtered = [
        (i, c)
        for i, c in enumerate(clips)
        if c["tense"] in selected_tenses
        and c.get("source_video", "unknown") in selected_sources
        and (search_query.lower() in c["sentence"].lower() if search_query else True)
    ]

    if not filtered:
        st.warning("No clips match your filters.")
        st.stop()

    for tense in selected_tenses:
        tense_clips = [(i, c) for i, c in filtered if c["tense"] == tense]
        if not tense_clips:
            continue

        label = TENSE_LABELS.get(tense, tense)
        color = TENSE_COLORS.get(tense, "#333")

        st.markdown(
            f"<h2 style='color:{color};'>{label} "
            f"<span style='font-size:0.6em;color:gray;'>({len(tense_clips)} clips)</span></h2>",
            unsafe_allow_html=True,
        )
        st.markdown(f"*{TENSE_EXPLANATIONS.get(tense, '')}*")

        per_page = 4
        total_pages = (len(tense_clips) - 1) // per_page + 1
        page_key = f"page_{tense}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1

        page = st.session_state[page_key]
        start = (page - 1) * per_page
        page_clips = tense_clips[start : start + per_page]

        for global_idx, clip in page_clips:
            video_path = clip["clip"].replace("\\", "/")
            sentence = clip["sentence"]
            timestamp = clip["time"]
            context = clip.get("context", [])
            enrich_key = f"enrich_{global_idx}"

            if st.session_state.get(enrich_key) == "running":
                wm = _whisper_model_arg()
                with st.spinner(
                    f"🎙️ Transcribing `{os.path.basename(video_path)}`… (~15 s)"
                ):
                    subprocess.run(
                        [
                            PYTHON,
                            "enrich_clips.py",
                            "--index",
                            str(global_idx),
                            "--whisper-model",
                            wm,
                        ],
                        capture_output=True,
                        text=True,
                        cwd=_APP_DIR,
                    )
                st.session_state[enrich_key] = "done"
                load_clips.clear()
                st.rerun()

            display_sentence = sentence
            if search_query:
                display_sentence = re.sub(
                    f"({re.escape(search_query)})",
                    r"**\1**",
                    sentence,
                    flags=re.IGNORECASE,
                )

            vcol, tcol = st.columns([5, 4], gap="large")

            with vcol:
                st.markdown(
                    f"<div style='background:{color}22; border-left:4px solid {color}; "
                    f"padding:8px 14px; border-radius:4px; margin-bottom:10px;'>"
                    f"<small style='color:gray;'>⏱ {timestamp} &nbsp;·&nbsp; "
                    f"<code style='font-size:0.8em;'>{os.path.basename(video_path)}</code></small><br>"
                    f"<span style='font-size:1.05em;'>{display_sentence}</span></div>",
                    unsafe_allow_html=True,
                )
                abs_video = str(Path(video_path).resolve())
                if os.path.exists(video_path):
                    st.video(abs_video)
                else:
                    st.warning(f"File not found: `{video_path}`")

                del_key = f"del_{global_idx}"
                if del_key not in st.session_state:
                    st.session_state[del_key] = False

                if not st.session_state[del_key]:
                    if st.button(
                        "🗑 Delete this clip",
                        key=f"del_btn_{global_idx}",
                        use_container_width=True,
                    ):
                        st.session_state[del_key] = True
                        st.rerun()
                else:
                    st.warning("Delete this clip permanently?")
                    dc1, dc2 = st.columns(2)
                    if dc1.button(
                        "✅ Yes", key=f"del_yes_{global_idx}", type="primary"
                    ):
                        if os.path.exists(video_path):
                            os.remove(video_path)
                        new_clips = [c for j, c in enumerate(clips) if j != global_idx]
                        save_clips(new_clips)
                        st.session_state[del_key] = False
                        st.rerun()
                    if dc2.button("❌ No", key=f"del_no_{global_idx}"):
                        st.session_state[del_key] = False
                        st.rerun()

            with tcol:
                st.markdown(
                    f"<div style='font-size:1.05em; font-weight:700; "
                    f"color:{color}; margin-bottom:8px; padding-bottom:4px; "
                    f"border-bottom:2px solid {color}44;'>📄 Transcript</div>",
                    unsafe_allow_html=True,
                )
                if context:
                    lines_html = ""
                    for seg in context:
                        t_label = f"{seg.get('start', 0):.1f}s"
                        seg_text = seg["text"]
                        if seg.get("highlight"):
                            lines_html += (
                                f"<div style='background:{color}44; border-left:3px solid {color}; "
                                f"padding:6px 12px; border-radius:3px; margin:4px 0;'>"
                                f"<div style='color:gray; font-size:0.78em; margin-bottom:2px;'>⏱ {t_label}</div>"
                                f"<strong style='color:white; font-size:0.97em;'>{seg_text}</strong>"
                                f"</div>"
                            )
                        else:
                            lines_html += (
                                f"<div style='padding:5px 12px; margin:2px 0;'>"
                                f"<div style='color:#666; font-size:0.78em;'>⏱ {t_label}</div>"
                                f"<span style='color:#bbb; font-size:0.95em;'>{seg_text}</span>"
                                f"</div>"
                            )
                    st.markdown(
                        f"<div style='max-height:340px; overflow-y:auto; "
                        f"border:1px solid {color}33; border-radius:6px; padding:4px 0;'>"
                        f"{lines_html}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='color:gray; font-style:italic; margin-top:12px;'>"
                        "No transcript yet for this clip.</p>",
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        "🎙️ Generate transcript",
                        key=f"gen_btn_{global_idx}",
                        type="primary",
                    ):
                        st.session_state[enrich_key] = "running"
                        st.rerun()

            st.markdown(
                "<hr style='margin:16px 0; opacity:0.15;'>", unsafe_allow_html=True
            )

        if total_pages > 1:
            pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
            with pcol1:
                if st.button("← Prev", key=f"prev_{tense}", disabled=page <= 1):
                    st.session_state[page_key] -= 1
                    st.rerun()
            with pcol2:
                st.markdown(
                    f"<p style='text-align:center; color:gray;'>Page {page} / {total_pages}</p>",
                    unsafe_allow_html=True,
                )
            with pcol3:
                if st.button(
                    "Next →", key=f"next_{tense}", disabled=page >= total_pages
                ):
                    st.session_state[page_key] += 1
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
