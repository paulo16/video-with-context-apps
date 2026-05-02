"""Unit tests for tense merge logic and spaCy-based classification."""

from extract_tenses import classify_tense, merge_clips_for_source


def test_merge_clips_replaces_same_source_only():
    existing = [
        {"source_video": "talk1", "sentence": "old", "clip": "clips/a.mp4"},
        {"source_video": "talk2", "sentence": "keep", "clip": "clips/b.mp4"},
    ]
    hits = [
        {"source_video": "talk1", "sentence": "new", "clip": "clips/c.mp4"},
    ]
    merged = merge_clips_for_source(existing, hits, "talk1")
    assert len(merged) == 2
    talk1 = [c for c in merged if c["source_video"] == "talk1"]
    assert len(talk1) == 1
    assert talk1[0]["sentence"] == "new"
    talk2 = [c for c in merged if c["source_video"] == "talk2"]
    assert len(talk2) == 1
    assert talk2[0]["sentence"] == "keep"


def test_classify_present_simple():
    t = classify_tense("She works every day.")
    assert "present_simple" in t


def test_classify_past_simple():
    t = classify_tense("She walked into the room.")
    assert "past_simple" in t


def test_classify_future_going_to():
    t = classify_tense("She is going to work tomorrow.")
    assert "future_going_to" in t


def test_classify_present_perfect():
    t = classify_tense("I have seen that movie.")
    assert "present_perfect" in t
