import re

from typing import List, Dict

# Minimal duplicate of synthesis logic to sanity check behavior without running Streamlit

def _similar(a: str, b: str) -> float:
    aw = {w for w in a.lower().split() if len(w) > 3}
    bw = {w for w in b.lower().split() if len(w) > 3}
    if not aw or not bw:
        return 0.0
    inter = len(aw & bw)
    union = len(aw | bw)
    return inter / union if union else 0.0


def synthesize(sentences: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for item in sentences:
        dup = False
        for keep in out:
            if _similar(str(item.get("text", "")), str(keep.get("text", ""))) > 0.85:
                if float(item.get("score", 0)) > float(keep.get("score", 0)):
                    keep.update(item)
                dup = True
                break
        if not dup:
            out.append(item)
    return out


def test_synthesis_merges_near_duplicates():
    s1 = {"text": "The convoy will stage at Gate Alpha with EOD overwatch.", "score": 1.6}
    s2 = {"text": "Convoy will stage at Gate Alpha with EOD overwatch in place.", "score": 1.3}
    s3 = {"text": "Medical triage at the reception zone is prioritized.", "score": 1.4}
    merged = synthesize([s1, s2, s3])
    # Expect 2 items after merging s1 and s2
    assert len(merged) == 2
    # Ensure higher score survives
    texts = [m["text"] for m in merged]
    assert any("The convoy will stage" in t for t in texts)


def test_similarity_threshold_sensible():
    a = "Immediate evacuation through Corridor A at 5 minute intervals"
    b = "Evacuation through Corridor B at 7 minute intervals with QRF"
    sim = _similar(a, b)
    assert 0.0 <= sim <= 1.0
