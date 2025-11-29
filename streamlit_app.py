import os
import io
import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

# Load .env first, then override with n.env if present (your project uses n.env)
load_dotenv()
if os.path.exists("n.env"):
    load_dotenv("n.env", override=True)
API_URL = os.getenv("API_URL", "http://localhost:5050/run")

def _api_url() -> str:
    return st.session_state.get("api_url", API_URL)

def _health_url(run_url: str) -> str:
    base = run_url.rstrip("/")
    if base.endswith("/run"):
        base = base[: -len("/run")]
    return base + "/health"

def _probe_backend(run_url: str, timeout: float = 1.5) -> bool:
    try:
        url = _health_url(run_url)
        r = requests.get(url, timeout=timeout)
        return r.ok
    except Exception:
        return False

def _autodetect_backend():
    # Avoid repeated checks per session
    if st.session_state.get("api_checked"):
        return
    candidates = []
    # User-set, env, common defaults
    if "api_url" in st.session_state:
        candidates.append(st.session_state["api_url"])
    if API_URL:
        candidates.append(API_URL)
    candidates += [
        "http://localhost:3001/run",
        "http://127.0.0.1:3001/run",
        "http://localhost:5050/run",
        "http://127.0.0.1:5050/run",
    ]
    # De-dup while preserving order
    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    for url in ordered:
        if _probe_backend(url):
            st.session_state["api_url"] = url
            st.session_state["api_checked"] = True
            try:
                st.toast(f"Connected to backend: {url}", icon="🟢")
            except Exception:
                pass
            return
    # If none responded, keep current but mark checked
    st.session_state.setdefault("api_url", API_URL)
    st.session_state["api_checked"] = True

# Optional structured formatting hint (disabled by default via sidebar toggle)
def _format_hint() -> str:
    if not st.session_state.get("force_template", False):
        return ""
    return (
        "\n\nFORMAT THE RESPONSE IN MARKDOWN USING CLEAR HEADINGS."
        " Write in fully developed paragraphs for each section; use lists only when they add clarity."
        " If the prompt specifies a structure, follow it; otherwise choose the most natural structure for the task.\n"
    )

def _extreme_detail_directive() -> str:
    # Switch behavior based on verbosity setting
    if not st.session_state.get("verbose_mode", False):
        return (
            "\n\nAnswer the USER REQUEST directly and concisely. Do not include methodology or meta explanations unless explicitly asked. "
            "Use short, clear paragraphs and only include lists or sections if the prompt requests them. Avoid boilerplate and stay on-topic."
        )
    return (
        "\n\nIMPORTANT: Your response MUST be extremely extensive and detailed, regardless of agent or prompt. "
        "Explain the approach you take, your step-by-step methodology, assumptions, constraints, trade-offs, and rationale. "
        "Include: Objectives, Assumptions, Constraints, Methodology/Approach (step-by-step), Alternatives and Trade-offs, Risks & Mitigations, Metrics & KPIs, Resources & Roles, Edge Cases & Failure Modes, Example Scenarios, and Concrete Next Actions. "
        "Expand acronyms on first use. Use paragraphs as the primary vehicle for explanation; include lists only to support complex enumerations. Use clear section headings. "
        "If documents are provided, ground claims in them and include short quotes where helpful; label any general knowledge add-ons as [External]. "
        "If web enrichment is enabled, integrate concise, relevant web findings with clear [Web:<domain>] source tags and do NOT fabricate URLs. "
        f"Target about {int(st.session_state.get('target_word_count', 1800))} words when possible. Be exhaustive and precise; do not be brief."
    )

def _web_use_allowed() -> bool:
    return bool(st.session_state.get("enable_web", False))

def _web_directive() -> str:
    if not _web_use_allowed():
        return "\n\nWeb browsing is DISABLED for this request; rely solely on provided documents and internal reasoning."
    depth = st.session_state.get("web_depth", "advanced")
    max_r = st.session_state.get("web_max_results", 6)
    return (
        f"\n\nWeb browsing is ENABLED (depth={depth}, max_results={max_r}). "
        "You may incorporate high-signal, recent, authoritative web snippets. Summarize them; cite with [Web:<domain>] and keep only directly relevant facts. "
        "Avoid paywalled, unverifiable, or speculative content; flag uncertainties explicitly."
    )

st.set_page_config(page_title="Agent", layout="wide")

# Sidebar: session
with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    # One-time backend autodetect (sets session_state.api_url if reachable)
    _autodetect_backend()
    # Allow manual override and quick test
    api_val = st.text_input("Backend URL", value=_api_url(), key="api_url_input")
    if api_val and api_val != st.session_state.get("api_url"):
        st.session_state["api_url"] = api_val
        st.session_state["api_checked"] = False
        _autodetect_backend()
    if st.button("Test backend"):
        ok = _probe_backend(_api_url())
        if ok:
            st.success(f"Backend OK: {_api_url()}")
        else:
            st.error(f"Backend not reachable: {_api_url()}")
    st.caption(f"Using backend: {_api_url()}")
    # Answer behavior controls
    st.toggle("Structured template", key="force_template", value=False, help="Force a rigid template. Leave off to let the model pick the best structure.")
    answer_mode = st.radio(
        "Answer mode",
        ["Auto", "Docs preferred", "Documents only"],
        index=1,
        help="How strongly to rely on uploaded documents.")
    st.session_state["answer_mode"] = answer_mode

    # Adjustable chat text size for readability
    chat_font_size = st.slider("Text size", min_value=16, max_value=24, value=19, step=1)
    st.session_state["chat_font_size"] = chat_font_size

    # Response time budget
    resp_budget = st.slider("Response time (s)", min_value=10, max_value=60, value=20, step=5, help="Upper limit for streaming per reply.")
    st.session_state["response_budget"] = resp_budget
    # Provenance + highlighting options
    st.toggle("Show source provenance", key="show_provenance", value=False,
              help="Append a short section listing which files contributed.")
    st.toggle("Highlight prompt keywords", key="highlight_keywords", value=False,
              help="Bold words from your prompt when they appear in the answer.")
    st.toggle("Show line numbers", key="show_line_numbers", value=False,
              help="Include approximate source line numbers for extracted evidence (local fallback only).")
    st.toggle("Show confidence scores", key="show_confidence", value=False,
              help="Append High/Medium/Low confidence labels to evidence lines (local fallback only).")
    # Target length & readability
    st.session_state["target_word_count"] = st.slider(
        "Target word count", min_value=800, max_value=20000, value=1800, step=100,
        help="Desired length hint passed to the model and used in local fallback expansion.")
    st.toggle("Show readability metrics", key="show_readability", value=False,
              help="Append Flesch score and other metrics for local fallback replies.")

    # Reply verbosity: keep OFF by default for concise, direct answers
    st.toggle(
        "Verbose answers",
        key="verbose_mode",
        value=False,
        help="When on, produce very detailed, methodology-rich answers. When off, respond concisely and directly to your prompt.",
    )

    # Group web controls in an expander to avoid overlapping bottom selector widgets
    with st.expander("Web Enrichment", expanded=False):
        st.toggle("Enable web browsing", key="enable_web", value=True,
                  help="Permit agents to perform web search enrichment for added precision (requires backend key).")
        st.session_state["web_depth"] = st.selectbox(
            "Web search depth", ["basic", "advanced"], index=1,
            help="basic = quicker; advanced = broader fetch scope.")
        st.session_state["web_max_results"] = st.slider(
            "Web max results", min_value=3, max_value=10, value=6, step=1,
            help="Maximum web results the backend may retrieve.")
    if st.button("Clear caches"):
        st.session_state.pop("doc_cache", None)
        try:
            st.toast("Caches cleared.", icon="🧹")
        except Exception:
            st.info("Caches cleared.")

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = []
# Remove legacy fallback state (no longer used)

# Render chat history (mark assistant bubbles for white styling) with re-run buttons
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        role = msg.get("role","assistant")
        content = msg.get("content","")
        frozen = msg.get("frozen_markdown")
        st.markdown(frozen or content)
        # Offer a re-run button for assistant messages that have an original prompt stored
        if role == "assistant" and msg.get("prompt"):
            if st.button("Re-run with current settings", key=f"rerun_{i}"):
                st.session_state["queued_prompt"] = msg["prompt"]
                st.rerun()

def _read_text_file(file, max_chars: int = 200_000_000) -> str:
    try:
        content = file.getvalue()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="ignore")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n[...truncated, {len(text)} chars total]"
        return text
    except Exception as e:
        return f"[Error reading {getattr(file, 'name', 'attachment')}: {e}]"

def _read_pdf_file(file, max_chars: int = 400_000) -> str:
    try:
        bio = io.BytesIO(file.getvalue())
        reader = PdfReader(bio)
        texts = []
        for page in reader.pages:
            # extract_text can return None; coalesce to empty string
            page_text = page.extract_text() or ""
            texts.append(page_text)
        text = "\n\n".join(texts)
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n[...truncated, {len(text)} chars total]"
        return text.strip()
    except Exception as e:
        return f"[Error reading PDF {getattr(file, 'name', 'attachment')}: {e}]"

def _split_into_chunks(text: str, max_chars: int = 12000, overlap: int = 500):
    """Split long text into overlapping chunks to fit model context more reliably."""
    text = text or ""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    end = max_chars
    while start < len(text):
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        end = start + max_chars
    return chunks

def _summarize_full_document(name: str, raw_text: str) -> str:
    """Ensure the ENTIRE PDF is evaluated by summarizing it chunk-by-chunk, then merging.
    Uses the backend overall_summary agent for consistency with current prompts.
    """
    chunks = _split_into_chunks(raw_text, max_chars=12000, overlap=500)
    if len(chunks) == 1:
        return raw_text

    # Map step: summarize each chunk
    part_summaries = []
    for i, ch in enumerate(chunks, start=1):
        directive = (
            f"You are processing PART {i} of {len(chunks)} of the document '{name}'. "
            "Summarize ALL salient facts and data points faithfully. Quote key lines. "
            "Do NOT include conclusions or recommendations. Keep this a detailed factual summary."
        )
        payload = (
            f"{directive}\n\n=== SOURCE EXCERPT (PART {i}/{len(chunks)}) ===\n{ch}\n=== END PART ==="
        )
        try:
            resp = requests.post(
                _api_url(),
                json={"input_as_text": payload, "target_agent": "overall_summary"},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            part_summaries.append(data.get("output_text", ""))
        except requests.RequestException as e:
            part_summaries.append(f"[Error summarizing part {i}: {e}]")

    # Reduce step: merge all part summaries into one doc summary
    combined = "\n\n".join(
        [f"### PART {i+1}\n{txt}" for i, txt in enumerate(part_summaries)]
    )
    merge_prompt = (
        "Merge the following PART summaries into a single, de-duplicated, coherent summary "
        f"of the document '{name}'. Use only the information present in the parts. "
        "No recommendations or conclusions."
    )
    try:
        resp = requests.post(
            _api_url(),
            json={
                "input_as_text": f"{merge_prompt}\n\n=== PART SUMMARIES ===\n{combined}\n=== END PART SUMMARIES ===",
                "target_agent": "overall_summary",
            },
            timeout=120,
        )
        resp.raise_for_status()
        merged = resp.json().get("output_text", combined)
    except requests.RequestException:
        merged = combined
    return merged

def _compose_input(user_text: str, files) -> str:
    files = files or []
    if files:
        # Build a strict, retrieval-style prompt to force using only provided sources
        src_blocks = []
        for f in files:
            name = getattr(f, "name", "attachment")
            fname = name.lower()
            if fname.endswith(".pdf"):
                raw = _read_pdf_file(f)
                # Ensure the entire document is evaluated: map-reduce summarize if large
                text = _summarize_full_document(name, raw)
            else:
                text = _read_text_file(f)
            src_blocks.append(f"### {name}\n{text}")

        directive = (
            "You MUST answer strictly and exclusively from the SOURCE DOCUMENTS below. "
            "Do not use any external knowledge. "
            "If the answer cannot be found verbatim or reasonably inferred from the sources, "
            "respond exactly with: Insufficient information in provided documents."
    ) + _extreme_detail_directive() + _web_directive()

        request = (user_text or "").strip()
        src_text = "\n\n".join(src_blocks)
        composed = (
            f"{directive}\n\n"
            f"=== SOURCE DOCUMENTS ===\n{src_text}\n=== END SOURCE DOCUMENTS ===\n\n"
            f"=== USER REQUEST ===\n{request}\n=== END USER REQUEST ==="
        )
        return (composed.strip() + _format_hint())
    # No files: fall back to plain message
    base = (user_text or "").strip()
    return (base + _extreme_detail_directive() + _web_directive() + _format_hint()) if base else base

def _compose_input_relaxed(user_text: str, files) -> str:
    """Compose a prompt that prefers provided files but allows external knowledge when needed.
    This avoids the hard failure message and provides a graceful fallback.
    """
    files = files or []
    if files:
        src_blocks = []
        for f in files:
            name = getattr(f, "name", "attachment")
            fname = name.lower()
            if fname.endswith(".pdf"):
                # Avoid network-heavy summarization here; just extract text quickly
                text = _read_pdf_file(f)
            else:
                text = _read_text_file(f)
            # Cap individual document text length sent to backend to avoid oversized payloads
            MAX_SEND = 60000
            if len(text) > MAX_SEND:
                text = text[:MAX_SEND] + f"\n[...truncated at {MAX_SEND} chars, original {len(text)} chars]"
            src_blocks.append(f"### {name}\n{text}")

        # Respect the selected answer mode
        mode = st.session_state.get("answer_mode", "Docs preferred")
        if mode == "Documents only":
            directive = (
                "You MUST answer strictly and exclusively from the SOURCE DOCUMENTS below. "
                "Do not use external knowledge unless explicitly asked to compare with outside information."
            ) + _extreme_detail_directive() + _web_directive()
        elif mode == "Auto":
            directive = (
                "Answer the USER REQUEST as clearly as possible. Use the SOURCE DOCUMENTS when relevant, but choose the structure that best fits the task."
            ) + _extreme_detail_directive() + _web_directive()
        else:  # Docs preferred
            directive = (
                "Use the SOURCE DOCUMENTS below as the primary basis for your answer. "
                "If key details are missing, you may add concise external knowledge, but keep it minimal and clearly relevant."
            ) + _extreme_detail_directive()
        request = (user_text or "").strip()
        src_text = "\n\n".join(src_blocks)
        composed = (
            f"{directive}\n\n"
            f"=== SOURCE DOCUMENTS ===\n{src_text}\n=== END SOURCE DOCUMENTS ===\n\n"
            f"=== USER REQUEST ===\n{request}\n=== END USER REQUEST ==="
        )
        return (composed.strip() + _format_hint())
    base = (user_text or "").strip()
    return (base + _extreme_detail_directive() + _web_directive() + _format_hint()) if base else base

def _sanitize_backend_reply(text: str) -> str:
    """Remove backend tool/run prefixes and other artifacts from replies."""
    if not text:
        return text
    import re
    t = text.strip()
    # Drop lines like '/run result type: string preview: ...' or other tool headers
    lines = []
    for line in t.splitlines():
        if re.match(r"\s*/?run result type:\s*", line, flags=re.I):
            continue
        if re.match(r"\s*tool result:\s*", line, flags=re.I):
            continue
        lines.append(line)
    t = "\n".join(lines).strip()
    # Strip stray backticks fences
    t = re.sub(r"```+", "", t)
    return t

def _stream_agent_response(body: dict, placeholder):
    """Attempt to stream backend response; fall back to full response then simulated streaming.
    Returns final reply text.
    """
    try:
        start = time.time()
        TIME_BUDGET = float(st.session_state.get("response_budget", 20))
        # First try real streaming (chunked transfer or SSE-like)
        resp = requests.post(_api_url(), json=body, timeout=TIME_BUDGET, stream=True)
        # If server rejects the payload (e.g., 413), surface friendly error and fallback
        if not resp.ok:
            try:
                err_preview = resp.text[:400]
            except Exception:
                err_preview = f"HTTP {resp.status_code}"
            placeholder.markdown(f"Server error: {err_preview}. Falling back locally…")
            return err_preview, False
        content_type = resp.headers.get("Content-Type", "")
        # If pure JSON (non-stream) just parse once
        if content_type.startswith("application/json"):
            data = resp.json()
            # Detect backend error shape and treat as failure -> let caller fallback
            if resp.status_code >= 400 or (isinstance(data, dict) and data.get("error")):
                err_text = data.get("error") or json.dumps(data, ensure_ascii=False)
                # Specific banner for missing key
                if "Missing OpenAI API key" in err_text:
                    placeholder.markdown("**Backend configuration error:** OpenAI API key is missing on the server. Set `OPENAI_API_KEY` in your `.env` or environment and restart the backend. Using local fallback.")
                else:
                    placeholder.markdown(f"Backend error: {err_text}. Falling back locally…")
                return err_text, False
            reply = data.get("output_text") or json.dumps(data, ensure_ascii=False)
            # Simulated streaming for UX, then render nicely formatted markdown at the end
            streamed = []
            for word in reply.split():
                streamed.append(word)
                if len(streamed) % 8 == 0:  # update more frequently for real-time feel
                    placeholder.markdown(" ".join(streamed))
                if time.time() - start > TIME_BUDGET:
                    break
            # Final pretty formatting pass
            final = _sanitize_backend_reply(" ".join(streamed))
            placeholder.markdown(_format_reply_markdown(final))
            return final, True
        # Real streaming: accumulate chunks and update
        chunks = []
        for raw in resp.iter_content(chunk_size=256, decode_unicode=True):
            if not raw:
                continue
            chunks.append(raw)
            # Update every chunk for smoother streaming
            placeholder.markdown("".join(chunks))
            if time.time() - start > TIME_BUDGET:
                break
        reply = "".join(chunks).strip()
        # Final pretty formatting pass on the full accumulated content
        final = _sanitize_backend_reply(reply or "(empty response)")
        # Detect HTML error payloads
        if "<html" in final.lower() or "payloadtoolargeerror" in final.lower():
            placeholder.markdown("Server returned an HTML error response. Falling back locally.")
            return final, False
        # If backend streamed an error block in JSON, handle it
        if final.strip().startswith("{") and "\"error\"" in final:
            return final, False
        if time.time() - start > TIME_BUDGET and not final:
            final = "Response truncated due to 20s limit."
        placeholder.markdown(_format_reply_markdown(final))
        return (final or ""), True
    except requests.RequestException as e:
        err = f"Request failed: {e}"
        placeholder.markdown(err)
        return err, False

def _format_reply_markdown(text: str) -> str:
    """Produce cleaner Markdown with headings and bullet points when possible.
    - If the text already contains markdown headings or lists, normalize spacing.
    - Otherwise, convert sentences to bullet points under a generic heading.
    """
    import re
    if not text:
        return ""
    t = text.strip()
    # Normalize excessive blank lines
    def _normalize(md: str) -> str:
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()

    # Normalize whitespace first
    t = t.replace("\r\n", "\n")
    t = re.sub(r"\t+", " ", t)

    # Ensure headings start on new lines
    t = re.sub(r"(?<!\n)(#{1,6}\s)", r"\n\n\1", t)

    # If a heading is followed by a dash used like a list starter, split it
    t = re.sub(r"^(#{2,6}\s[^\n]+?)\s+-\s", r"\1\n\n- ", t, flags=re.M)

    # Convert inline bullets like " - " into real line bullets
    t = re.sub(r"(\S)\s-\s", r"\1\n- ", t)

    # Make sure list items begin on lines
    t = re.sub(r"(?<!\n)-\s", r"\n- ", t)

    # Convert bullet characters to markdown dashes and line them up
    t = re.sub(r"(\S)\s•\s", r"\1\n- ", t)
    t = re.sub(r"^•\s", r"- ", t, flags=re.M)

    # Collapse excessive blank lines
    t = _normalize(t)

    # If it now contains structured cues, return as-is unless the template is forced
    if any(h in t for h in ("\n# ", "\n## ", "\n### ", "\n- ", "\n* ")):
        return _ensure_fixed_sections(t) if st.session_state.get("force_template") else t

    # No clear markdown structure: create coherent paragraphs instead of bullets
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", t)
    sents = [s.strip() for s in sentences if s.strip()]
    if not sents:
        return _normalize(t)
    paras = []
    buf = []
    for s in sents:
        buf.append(s)
        if len(buf) >= 3:
            paras.append(" ".join(buf))
            buf = []
    if buf:
        paras.append(" ".join(buf))
    body = "\n\n".join(paras)
    fallback = f"## Narrative\n\n{body}"
    return _ensure_fixed_sections(fallback) if st.session_state.get("force_template") else fallback

def _ensure_fixed_sections(md: str) -> str:
    """Ensure the reply has the fixed section order and headers.
    If missing, create placeholders. Keeps existing content but organizes it.
    """
    import re
    md = md.strip()
    # Extract a reasonable title (first heading or first line)
    title = "Title"
    m = re.search(r"^#{1,6}\s+(.+)$", md, flags=re.M)
    if m:
        title = m.group(1).strip()
    else:
        first_line = md.splitlines()[0] if md else ""
        title = first_line.strip()[:80] or "Title"

    def section(name: str) -> str:
        pattern = rf"^###\s+{re.escape(name)}\b"
        return pattern

    # Ensure each section header exists; if not, append it
    sections = [
        "Situation Overview",
        "Key Points",
        "Risks and Constraints",
        "Recommended Actions",
    ]

    out = []
    out.append(f"## {title}")

    lower = md.lower()
    for name in sections:
        if re.search(section(name), md, flags=re.M):
            # Extract existing section content
            # Capture from this header to next header or end
            sec_re = rf"^###\s+{re.escape(name)}\b[\s\S]*?(?=^###\s+|^##\s+|\Z)"
            m = re.search(sec_re, md, flags=re.M)
            out.append(m.group(0).strip() if m else f"### {name}\n\n- None.")
        else:
            out.append(f"### {name}\n\n- None.")
    return "\n\n".join(out)

def _render_thinking_skeleton(placeholder):
    """Lightweight placeholder while work is in progress."""
    placeholder.markdown("Thinking…")

def _extract_key_sentences(user_text: str, doc_text: str, max_sentences: int = 24):
    import re
    text = (doc_text or "").replace("\r\n", "\n")
    # Rough sentence split
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text)
    # Keyword set from user prompt
    kws = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", user_text or "")}
    # Precompute line breaks for line number mapping
    line_offsets = []
    acc = 0
    for line in text.splitlines(keepends=True):
        line_offsets.append(acc)
        acc += len(line)
    def line_number_for_pos(pos: int) -> int:
        # Binary search could be used; linear is fine for typical sizes
        ln = 1
        for i, off in enumerate(line_offsets):
            if off > pos:
                break
            ln = i + 1
        return ln
    def score(s: str) -> float:
        s_clean = s.strip()
        if not s_clean:
            return 0
        length = len(s_clean)
        if length < 40:  # too short to be informative
            return 0.2
        if length > 400:  # too long
            return 0.3
        has_digit = any(c.isdigit() for c in s_clean)
        caps = sum(1 for t in s_clean.split() if t.isupper() and len(t) > 1)
        kw_overlap = sum(1 for w in re.findall(r"[A-Za-z]{4,}", s_clean.lower()) if w in kws)
        punctuation = s_clean.count(";") + s_clean.count(":") + s_clean.count(",")
        return 1.0 + 0.5*has_digit + 0.1*caps + 0.2*kw_overlap + 0.05*punctuation
    ranked = sorted(sentences, key=score, reverse=True)
    # Keep unique-ish sentences
    seen = set()
    picked = []
    for s in ranked:
        k = s.strip()[:80].lower()
        if not k or k in seen:
            continue
        seen.add(k)
        # Map to line number range (approx) by locating first occurrence
        try:
            pos = text.index(s)
        except ValueError:
            pos = 0
        start_line = line_number_for_pos(pos)
        end_line = line_number_for_pos(pos + len(s))
        picked.append({"text": s.strip(), "score": score(s), "lines": (start_line, end_line)})
        if len(picked) >= max_sentences:
            break
    return picked

def _file_hash(file) -> str:
    try:
        import hashlib
        content = file.getvalue()
        return hashlib.md5(content).hexdigest()
    except Exception:
        return getattr(file, "name", "attachment")

def _get_cached_sentences(user_text: str, file, max_sentences: int = 30):
    """Cache extracted sentences per file content hash to avoid repeat work."""
    cache = st.session_state.setdefault("doc_cache", {})
    fh = _file_hash(file)
    key = (fh, max_sentences, tuple(sorted(set([w.lower() for w in (user_text or "").split()]))) )
    if key in cache:
        return cache[key]
    # Compute
    name = getattr(file, "name", "attachment")
    try:
        if name.lower().endswith('.pdf'):
            txt = _read_pdf_file(file, max_chars=120000)
        else:
            txt = _read_text_file(file, max_chars=120000)
    except Exception:
        txt = ""
    sentences = _extract_key_sentences(user_text, txt, max_sentences=max_sentences) if txt else []
    cache[key] = sentences
    return sentences

def _clean_sentences(sentences, max_n: int = 24):
    """Dedupe and filter noisy lines (codes, garbled chars, very short). Preserve order."""
    import re
    seen = set()
    clean = []
    for item in sentences or []:
        # item may be dict from _extract_key_sentences
        if isinstance(item, dict):
            s0 = " ".join((item.get("text", "") or "").split()).strip()
            score = item.get("score", 0.0)
            lines = item.get("lines", (None, None))
        else:
            s0 = " ".join((item or "").split()).strip()
            score = 0.0
            lines = (None, None)
        if not s0:
            continue
        # Filter noisy patterns
        if re.search(r"[■�█]", s0):
            continue
        if re.match(r"^\(\d{2,3}\)\s", s0):
            continue
        if len(s0) < 40:
            continue
        key = s0[:90].lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append({"text": s0, "score": score, "lines": lines})
        if len(clean) >= max_n:
            break
    return clean

def _generate_local_fallback(user_text: str, files) -> str:
    """Produce a detailed local reply that adapts to the prompt and mines PDFs/texts for specifics."""
    files = files or []
    extracted = []
    provenance = []
    # Track per-sentence provenance for optional display
    sentence_sources = []
    for f in files:
        name = getattr(f, "name", "attachment")
        sents = _get_cached_sentences(user_text, f, max_sentences=60)
        extracted.extend(sents)
        if sents:
            provenance.append((name, len(sents)))
            for s in sents:
                if isinstance(s, dict):
                    sentence_sources.append({"file": name, "text": s.get("text"), "score": s.get("score"), "lines": s.get("lines")})
                else:
                    sentence_sources.append({"file": name, "text": s, "score": 0.0, "lines": (None, None)})
    extracted = _clean_sentences(extracted, max_n=28)

    # Simple multi-document synthesis: remove near-duplicates across different files keeping highest score
    def _similar(a: str, b: str) -> float:
        aw = {w for w in a.lower().split() if len(w) > 3}
        bw = {w for w in b.lower().split() if len(w) > 3}
        if not aw or not bw:
            return 0.0
        inter = len(aw & bw)
        union = len(aw | bw)
        return inter / union if union else 0.0
    synthesized = []
    for item in extracted:
        dup = False
        for keep in synthesized:
            if _similar(item["text"], keep["text"]) > 0.85:
                # keep the one with higher score
                if item.get("score", 0) > keep.get("score", 0):
                    keep.update(item)
                dup = True
                break
        if not dup:
            synthesized.append(item)
    extracted = synthesized

    # Map confidence labels
    def _confidence(score: float) -> str:
        if score >= 1.6:
            return "High"
        if score >= 1.3:
            return "Medium"
        return "Low"

    prompt = (user_text or "").strip()
    p_lower = prompt.lower()
    # Intent-based structure
    if any(w in p_lower for w in ["civilian", "civilians", "evacuate", "exfil", "extract", "rescue", "safe corridor"]):
        title = "Civilian Extraction Decision Brief"
        # Use extracted as supporting observations only; generate doctrine-oriented content
        observations = [e["text"] for e in extracted[:10]]
        sections = {
            "Mission": [
                "Safely locate, consolidate, and extract civilians from contested urban areas to designated safe zones while minimizing exposure to indirect fire, small-arms, and IED hazards.",
                "Maintain positive identification (PID) of civilians and preserve medical and logistics capacity for sustained movement."],
            "Situation": [
                "Urban terrain with potential subterranean access points, booby-trapped entryways, and line-of-sight sniper positions.",
                "Intermittent communications and UAV coverage; risk of decoys and secondary devices at choke points."],
            "Assumptions": [
                "Civilians may be sheltering in place and unable to self-evacuate.",
                "Limited windows of reduced contact permit movement along pre‑cleared corridors."],
            "End State": [
                "All civilians registered, screened, medically triaged, and transported to safe zones with accountability maintained (100%)."] ,
            "Courses of Action": [
                "COA‑A (Corridor + Convoy): Establish two parallel safe corridors with overwatch and layered EOD; move civilians in serials using marked vehicles and ambulances.",
                "COA‑B (Hub‑and‑Spoke): Create local assembly hubs (schools/clinics), then shuttle to a central transfer site; reduces exposure per movement but requires more staging security.",
                "COA‑C (Night Extraction + UAV Illumination): Use low‑light window with IR marking; demands tight comms discipline and rehearsals."],
            "Evaluation": [
                "COA‑A: Fastest egress; higher predictability → higher ambush/IED risk unless corridors are frequently varied.",
                "COA‑B: Best control/accountability; requires sustained security at hubs and medical support on site.",
                "COA‑C: Lower visual signature; navigation/ID errors more likely; reserve for low traffic periods after rehearsal."],
            "Recommended COA": [
                "COA‑B with an initial COA‑A pilot for time‑on‑route validation, then switch to hub‑and‑spoke with rolling QRF and EOD leapfrogging."],
            "Phased Execution": [
                "Phase 0 – Prep: Publish NOTAM/NOTICE to local leaders; distribute instructions via SMS/leaflets; stage medics, buses, water, and lighting.",
                "Phase 1 – Corridor Clearance: EOD/EOR teams clear 200m bounds; mark hazards; UAV/rooftop overwatch established.",
                "Phase 2 – Consolidation: Teams register civilians, segregate by medical priority, issue wristbands/badges; baggage screening.",
                "Phase 3 – Movement: Serialize convoys (lead EOD, bus/ambulance, trail QRF); stagger at 5–7 min intervals; checkpoints log counts.",
                "Phase 4 – Reception: Triage on arrival; reunification area; debrief and update accountability ledger."],
            "Risk / Controls": [
                "IED/Booby Traps → EOD lead, ground‑truthing mirrors/hooks, doorway tape‑wire checks, robot where feasible.",
                "Sniper/Harassing Fire → Rooftop shields, smoke screens, bounding overwatch, no static queues.",
                "Crowd Surge → Barriers, marshals, loudhailers, pre‑brief in multiple languages, water points every 150m.",
                "Comms Loss → Simple color/hand signals, backup runners, pre‑agreed triggers (e.g., red flare = hold; green = continue)."],
            "Coordination": [
                "Assign a Movement Control Team (MCT) and a Civilian Protection Officer (CPO) per hub.",
                "Deconflict with NGOs/UN for shelter capacity, food/water, and family tracing.",
                "ROE/Legal: PID standards reiterated; protected status signage; detainee handling lane if needed."],
            "Logistics/Medical": [
                "Ambulances at tail/center of serial; medical triage tags (MARCH/START).",
                "Spare bus, fuel cache every 3–4 km; portable lighting; AED kits; stretchers; pediatric kits."],
            "Triggers & Contingencies": [
                "Abort if two consecutive checkpoints report loss of comms > 3 min or confirmed IED ahead not cleared in 10 min.",
                "Alternate routes per sector; safe‑haven fallback sites pre‑briefed; family reunification protocol active."],
        }
        if observations:
            sections["Supporting Observations"] = [f"\"{o}\"" for o in observations]
    elif any(w in p_lower for w in ["action plan", "execute", "implementation", "roadmap"]):
        title = "Action Plan"
        sections = {
            "Objectives": [e["text"] for e in extracted[:5]] or ["Clarify outcomes, scope, and constraints."],
            "Phases": [e["text"] for e in extracted[5:10]] or ["Phase 1: Discovery", "Phase 2: Build", "Phase 3: Validate", "Phase 4: Deploy"],
            "Key Tasks": [e["text"] for e in extracted[10:20]] or ["Define tasks, owners, and due dates.", "Sequence dependencies.", "Prepare runbooks."],
            "Risks": ["Resourcing limits", "Timeline slippage", "Stakeholder misalignment", "Data quality issues"],
            "Metrics": ["On-time delivery", "Quality acceptance rate", "Risk burndown", "Stakeholder satisfaction"],
            "Next 72 Hours": extracted[20:30] or ["Kickoff meeting", "Finalize plan & owners", "Stand up tracking board"],
        }
    elif any(w in p_lower for w in ["summary", "summarize", "synopsis", "brief"]):
        title = "Executive Summary"
        sections = {
            "Key Findings": [e["text"] for e in extracted[:12]] or ["No salient findings extracted from documents."],
            "Evidence/Quotes": [e["text"] for e in extracted[12:24]] or ["No quotations available."],
            "Implications": ["Highlight operational or decision impacts.", "Flag critical gaps or dependencies."],
            "Recommendations": ["Prioritize high-impact items.", "Define immediate next steps."],
        }
    elif any(w in p_lower for w in ["decision", "choose", "should we", "recommendation"]):
        title = "Decision Brief"
        sections = {
            "Options": [e["text"] for e in extracted[:10]] or ["Option A", "Option B"],
            "Analysis": [e["text"] for e in extracted[10:20]] or ["Relative benefits, costs, risks."],
            "Recommendation": ["Choose the option that maximizes outcomes under constraints; specify rationale."],
            "Execution Steps": extracted[20:30] or ["Define owners, timeline, success criteria."],
        }
    else:
        title = "Answer"
        sections = {
            "Details": [e["text"] for e in extracted[:14]] or ["No document-derived details available."],
            "Context": [e["text"] for e in extracted[14:22]] or ["Provide assumptions and boundary conditions."],
            "Next Steps": [e["text"] for e in extracted[22:30]] or ["Outline immediate actions and owners."],
        }

    # Build markdown
    parts = []
    # In concise mode, skip templated title unless evidence exists
    concise_mode = not st.session_state.get("verbose_mode", False)
    if not concise_mode:
        parts.append(f"## {title}")
    # Include methodology section only when verbose mode is enabled
    if st.session_state.get("verbose_mode", False):
        used_docs = ", ".join(n for n, _ in provenance) if provenance else "None"
        parts.append(
            "### Methodology & Approach\n"
            f"This answer was produced by extracting and ranking salient sentences from provided documents (if any), weighting informativeness such as presence of numerics, punctuation density, and overlap with your prompt. Noisy or duplicated lines were removed, and near-duplicates across documents were synthesized using token-set similarity, keeping the highest-confidence variant. Where possible, approximate line numbers are included for traceability back to the source text. Confidence labels (High/Medium/Low) reflect heuristic scoring. Documents considered: {used_docs}."
        )
    # Add a paragraph-style narrative synthesis from extracted evidence
    def _paragraphs_from_evidence(items, min_para_len=3):
        texts = [it.get("text") if isinstance(it, dict) else str(it) for it in items]
        texts = [t for t in texts if t]
        paras, buf = [], []
        for t in texts:
            buf.append(t)
            if len(buf) >= min_para_len:
                paras.append(" ".join(buf))
                buf = []
        if buf:
            paras.append(" ".join(buf))
        return paras[:6]  # keep the top few paragraphs for readability

    def _merge_short_paragraphs(paras, min_len=120):
        merged = []
        buf = ""
        for p in paras:
            p = (p or "").strip()
            if not p:
                continue
            if len(p) < min_len:
                if buf:
                    buf = f"{buf} {p}".strip()
                else:
                    buf = p
                continue
            if buf:
                merged.append(buf)
                buf = ""
            merged.append(p)
        if buf:
            merged.append(buf)
        return merged
    narrative = []
    if st.session_state.get("verbose_mode", False):
        narrative = _paragraphs_from_evidence(extracted, min_para_len=3)
        # Merge consecutive short narrative paragraphs to avoid choppiness
        narrative = _merge_short_paragraphs(narrative, min_len=120)
        if narrative:
            parts.append("### Detailed Narrative\n" + "\n\n".join(narrative))
    # Track used evidence texts to enable section balancing
    used_texts = set()
    for p in narrative:
        used_texts.update({p})
    # Build sections only in verbose mode OR when we have real extracted evidence
    if not concise_mode:
        for name, items in sections.items():
            if items:
                para = " ".join(str(i) for i in items)
                if len(para) < 280:
                    unused = [it.get("text") if isinstance(it, dict) else str(it) for it in extracted if (it.get("text") if isinstance(it, dict) else str(it)) not in used_texts]
                    add_buf = []
                    for s in unused:
                        if not s:
                            continue
                        add_buf.append(s)
                        if len(para + " " + " ".join(add_buf)) >= 320:
                            break
                    if add_buf:
                        para = (para + " " + " ".join(add_buf)).strip()
                        used_texts.update(add_buf)
                parts.append(f"### {name}\n{para}")

    # Concise mode fallback: direct answer without template noise
    if concise_mode:
        if extracted:
            direct = " ".join([e["text"] if isinstance(e, dict) else str(e) for e in extracted[:6]])
            return _format_reply_markdown(direct)
        # If no evidence and no backend, generate a minimal direct response rather than echoing prompt
        if user_text:
            prompt = user_text.strip()
            # Simple heuristic: if prompt looks like a question and lacks a '?', add clarifying sentence
            if not prompt.endswith('?') and len(prompt.split()) <= 8:
                answer = f"Direct response: {prompt} — please provide more specific details to get a richer answer."
            else:
                answer = f"Direct response: {prompt}" if len(prompt) < 140 else prompt[:140] + "…"
        else:
            answer = "No input provided. Enter a prompt to receive an answer."
        return _format_reply_markdown(answer)
    # Optional detailed provenance with line numbers & confidence
    if st.session_state.get("show_provenance") and provenance:
        prov_lines = "\n".join(f"- {n}: {c} supporting sentences" for n,c in provenance)
        parts.append(f"### Sources Used\n{prov_lines}")
        if st.session_state.get("show_line_numbers"):
            detail_lines = []
            for item in extracted[:20]:  # limit for brevity
                ln = item.get("lines", (None, None))
                line_part = ""
                if ln[0]:
                    if ln[0] == ln[1]:
                        line_part = f"L{ln[0]}"
                    else:
                        line_part = f"L{ln[0]}-{ln[1]}"
                conf_part = ""
                if st.session_state.get("show_confidence"):
                    conf_part = f" ({_confidence(item.get('score',0.0))} confidence)"
                detail_lines.append(f"- {item['text'][:140]}" + (f" [{line_part}]" if line_part else "") + conf_part)
            if detail_lines:
                parts.append("### Evidence Line Mapping\n" + "\n".join(detail_lines))
    md = "\n\n".join(parts)
    out = _format_reply_markdown(md)
    # Append readability metrics if enabled
    if st.session_state.get("show_readability"):
        raw_text = _strip_markdown(out)
        metrics = _readability_metrics(raw_text)
        flesch = f"{metrics['flesch']:.1f}"
        grade = f"{metrics['fk_grade']:.1f}"
        extra = (
            "\n\n### Readability Metrics\n"
            f"Words: {metrics['words']} | Sentences: {metrics['sentences']} | Avg words/sentence: {metrics['avg_words_per_sentence']:.1f} | "
            f"Avg syllables/word: {metrics['avg_syllables_per_word']:.2f} | Flesch Reading Ease: {flesch} | Flesch-Kincaid Grade: {grade}"
        )
        out += extra
    # Optional keyword highlighting
    if st.session_state.get("highlight_keywords"):
        out = _highlight_keywords(user_text, out)
    return out if not st.session_state.get("force_template") else _ensure_fixed_sections(out)

def _highlight_keywords(user_text: str, md: str) -> str:
    """Bold prompt keywords (>=4 letters) in the markdown outside code blocks."""
    import re
    if not user_text or not md:
        return md
    kws = sorted({w.lower() for w in re.findall(r"[A-Za-z]{4,}", user_text)}, key=len, reverse=True)
    if not kws:
        return md
    # Split into code and non-code segments
    segments = re.split(r"(```[\s\S]*?```)", md)
    def repl(seg: str) -> str:
        if seg.startswith("```"):
            return seg
        s = seg
        for w in kws:
            # word boundary replace, case-insensitive
            s = re.sub(rf"(?i)\b({re.escape(w)})\b", r"**\1**", s)
        return s
    return "".join(repl(s) for s in segments)

def _stream_text_simulated(text: str, placeholder, time_budget: float = 8.0):
    start = time.time()
    words = text.split()
    out = []
    for w in words:
        out.append(w)
        if len(out) % 10 == 0:
            placeholder.markdown(" ".join(out))
        if time.time() - start > time_budget:
            break
    placeholder.markdown(_format_reply_markdown(" ".join(out)))

# --- Readability metrics (optional, local fallback only) ---
def _strip_markdown(md: str) -> str:
    import re
    if not md:
        return ""
    s = md
    # Remove code fences
    s = re.sub(r"```[\s\S]*?```", " ", s)
    # Remove inline code backticks
    s = s.replace("`", " ")
    # Replace links/images with their alt/text
    s = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", s)
    # Drop markdown headers, emphasis, and formatting marks
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.M)
    s = re.sub(r"[*_]{1,3}", "", s)
    # Remove HTML tags if any
    s = re.sub(r"<[^>]+>", " ", s)
    return s

def _count_syllables(word: str) -> int:
    w = word.lower()
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    # Subtract a trailing silent 'e'
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def _readability_metrics(text: str) -> dict:
    import re
    if not text:
        return {
            "words": 0, "sentences": 0, "syllables": 0,
            "avg_words_per_sentence": 0.0, "avg_syllables_per_word": 0.0,
            "flesch": 0.0, "fk_grade": 0.0,
        }
    # Sentence split similar to elsewhere in the app
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    # Word tokens
    words = re.findall(r"[A-Za-z']+", text)
    wc = len(words)
    sc = len(sentences) if sentences else 1
    if wc == 0:
        wc = 1
    syllables = sum(_count_syllables(w) for w in words) if words else 0
    avg_wps = wc / sc
    avg_spw = (syllables / wc) if wc else 0.0
    flesch = 206.835 - 1.015 * avg_wps - 84.6 * avg_spw
    fk_grade = 0.39 * avg_wps + 11.8 * avg_spw - 15.59
    return {
        "words": len(words),
        "sentences": len(sentences),
        "syllables": syllables,
        "avg_words_per_sentence": avg_wps,
        "avg_syllables_per_word": avg_spw,
        "flesch": flesch,
        "fk_grade": fk_grade,
    }

# Inject CSS to shrink and position the uploader like an icon button (to the left of the chat bar)
# Expose CSS variable for dynamic font sizing
st.markdown(
    f"""
    <style>
    :root {{ --chat-font-size: {st.session_state.get('chat_font_size', 19)}px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    :root { --chat-width: 75vw; --agent-width: 160px; --uploader-size: 28px; --gap: 16px; --chat-height: 56px; }

    /* Dark theme like ChatGPT */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"],
    [data-testid="stMainBlockContainer"], .main, .block-container,
    [data-testid="stHeader"], footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        background: #0b0f12 !important; color: #e5e7eb !important;
    }
    /* Sidebar dark */
    [data-testid="stSidebar"] {
        background: #0b0f12 !important; color: #e5e7eb !important; box-shadow: none !important;
    }
    [data-testid="stSidebar"] * { color: #e5e7eb !important; }
    [data-testid="stSidebar"] a { color: #e5e7eb !important; }
    /* Sidebar buttons: dark background with light text */
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] button {
        background: #111827 !important;
        color: #e5e7eb !important;
        border: 1px solid #374151 !important;
        box-shadow: none !important;
    }
    /* White bottom safety band (fallback) */
    /* Remove previous white safety band and unify dark background */
    body::after { display: none !important; }
    /* Remove any bottom border/line or shadow some themes add */
    [data-testid="stAppViewContainer"] .main,
    [data-testid="stAppViewContainer"] .main .block-container,
    footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
        border: none !important; box-shadow: none !important;
    }
    hr { border-color: #111827 !important; background: #111827 !important; color: #111827 !important; height: 0 !important; }
    div, p, span, label { color: #e5e7eb !important; }

    /* High-contrast inputs like ChatGPT */
    input, textarea { background: #111827 !important; color: #e5e7eb !important; }
    input::placeholder, textarea::placeholder { color: #9ca3af !important; }
    button { color: #e5e7eb !important; }

    /* Style the native chat input and pin it to bottom center */
    [data-testid="stChatInput"] {
        position: fixed !important;
        left: 50% !important; transform: translateX(-50%) !important;
        bottom: 32px !important; z-index: 1200 !important;
        width: min(760px, 88vw) !important; 
        min-width: 360px !important;
        background: transparent !important; border: none !important; box-shadow: none !important;
        padding: 0 !important; margin: 0 !important;
    }
    [data-testid="stChatInput"] * { color: #e5e7eb !important; }
    /* Eliminate darker frame/wrapper around the input */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="textarea"]::before,
    [data-testid="stChatInput"] [data-baseweb="textarea"]::after {
        background: transparent !important; border: 0 !important; box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] div[contenteditable="true"],
    [data-testid="stChatInput"] div[role="textbox"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] textarea {
        background: #1f2937 !important; color: #f1f5f9 !important;
        border: 1px solid #374151 !important; border-radius: 28px !important;
        height: 62px !important; padding: 0 22px 0 22px !important; font-size: 17px !important;
        box-shadow: none !important; outline: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder { color: #9ca3af !important; }

    /* Set main container to ~3/4 of viewport width and center it */
    /* Center chat column similar to ChatGPT: narrower, centered, generous vertical breathing room */
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 860px; /* narrower width */
        margin-left: auto !important; margin-right: auto !important;
        padding-left: 1.25rem; padding-right: 1.25rem;
        display: flex; flex-direction: column; align-items: center; /* hard-center all content */
    }
    /* Constrain chat messages to narrower readable width */
    div[data-testid="stChatMessageContent"] {
        max-width: 820px;
        margin-left: auto; margin-right: auto;
    }
    /* Center chat rows and constrain message container */
    div[data-testid="stChatMessage"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    div[data-testid="stChatMessage"] > div:last-child {
        flex: 0 1 840px !important;
        max-width: 840px !important;
    }
    /* Single width rule above keeps things consistent; no duplicate overrides */

    /* Hide chat avatars permanently (robust selectors) and remove reserved space */
    div[data-testid="stChatMessageAvatar"],
    div[data-testid*="ChatMessageAvatar"],
    div[data-testid*="chatMessageAvatar"],
    div[aria-label*="avatar" i],
    img[alt*="avatar" i],
    svg[aria-label*="avatar" i] {
        display: none !important;
        width: 0 !important; min-width: 0 !important; height: 0 !important;
    }
    /* If the row has an avatar cell as first child, hide it */
    div[data-testid="stChatMessage"] > div:first-child { display: none !important; width: 0 !important; }
    div[data-testid="stChatMessage"] { padding-left: 0 !important; margin-left: 0 !important; }
    div[data-testid="stChatMessageContent"] { margin-left: 0 !important; }

    /* Reset selects in the sidebar to normal flow (do not affect main content selects) */
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        position: static !important;
    }

    /* Force the main agent selectbox (first selectbox in the main column) to be fixed bottom-left */
    [data-testid="stAppViewContainer"] .main .block-container div[data-testid="stSelectbox"]:first-of-type {
        position: fixed !important;
        left: 24px !important; bottom: 32px !important; z-index: 2500 !important;
        width: auto !important; max-width: 180px !important; min-width: 110px !important;
    }
    [data-testid="stAppViewContainer"] .main .block-container div[data-testid="stSelectbox"]:first-of-type [aria-haspopup="listbox"] {
        background: #2b3341 !important; border: 1px solid #3b4352 !important;
        border-radius: 8px !important; height: 40px !important; padding: 0 12px !important;
        display: flex !important; align-items: center !important; overflow: hidden !important;
        text-overflow: ellipsis !important; color: #e5e7eb !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.35) !important;
    }
    [data-testid="stAppViewContainer"] .main .block-container div[data-testid="stSelectbox"]:first-of-type svg { display: none !important; }
    [data-testid="stAppViewContainer"] .main .block-container div[data-testid="stSelectbox"]:first-of-type input {
        opacity: 0 !important; width: 1px !important; min-width: 1px !important;
        padding: 0 !important; margin: 0 !important; border: 0 !important; background: transparent !important;
        color: transparent !important; caret-color: transparent !important;
    }
    /* Agent selector: fixed container; JS will keep it just to the left of the input */
    #agent-picker-fixed {
        position: fixed !important; left: 24px; bottom: 32px; z-index: 2499;
        min-width: 110px !important; max-width: 180px !important; width: fit-content !important;
        white-space: nowrap !important;
        pointer-events: auto !important;
    }
    #agent-picker-fixed div[data-baseweb="select"],
    #agent-picker-fixed div[data-baseweb="select"] *,
    #agent-picker-fixed div[data-baseweb="select"] [aria-haspopup="listbox"],
    #agent-picker-fixed div[data-baseweb="select"] div[role="combobox"],
    #agent-picker-fixed div[data-baseweb="select"] input {
        background: #1f2937 !important; color: #e5e7eb !important;
    }
    #agent-picker-fixed div[data-baseweb="select"] [aria-haspopup="listbox"] {
        background: #2b3341 !important; /* bluish gray like screenshot */
        border: 1px solid #3b4352 !important; border-radius: 8px !important;
        height: 40px !important; display: flex; align-items: center; padding: 0 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.35);
        overflow: hidden !important; text-overflow: ellipsis !important;
    }

    /* Hard-target the agent select by key id if relocation misses */
    div[id*="agent_selector"][data-testid="stSelectbox"] {
        position: fixed !important; left: 24px !important; bottom: 32px !important; z-index: 2499 !important;
        width: auto !important; max-width: 180px !important; min-width: 110px !important;
    }
    div[id*="agent_selector"][data-testid="stSelectbox"] [aria-haspopup="listbox"] {
        background: #2b3341 !important; border: 1px solid #3b4352 !important;
        border-radius: 8px !important; height: 40px !important; padding: 0 12px !important;
        display: flex !important; align-items: center !important; overflow: hidden !important; text-overflow: ellipsis !important;
        color: #e5e7eb !important; box-shadow: 0 1px 3px rgba(0,0,0,0.35) !important;
    }
    div[data-baseweb="popover"] *, div[role="listbox"], div[role="listbox"] * {
        background: #0f172a !important; color: #e5e7eb !important;
    }
    #agent-picker-fixed div[data-baseweb="select"] [aria-haspopup="listbox"] * {
        font-size: 13px !important; line-height: 18px !important; color: #e5e7eb !important;
    }
    /* Remove dropdown chevron for agent picker only */
    #agent-picker-fixed div[data-baseweb="select"] svg { display: none !important; }
    /* Keep inner input present for accessibility & interactions, but visually hide it */
    #agent-picker-fixed div[data-baseweb="select"] input {
        opacity: 0 !important; width: 1px !important; min-width: 1px !important;
        padding: 0 !important; margin: 0 !important; border: 0 !important;
        background: transparent !important; color: transparent !important; caret-color: transparent !important;
    }
    #agent-picker-fixed div[data-baseweb="select"] button[aria-label*="clear" i],
    #agent-picker-fixed div[data-baseweb="select"] [aria-label*="clear" i] { display: none !important; }
    /* Remove any borders/shadows on the value container */
    #agent-picker-fixed div[data-baseweb="select"] [aria-haspopup="listbox"] { outline: none !important; box-shadow: none !important; }
    #agent-picker-fixed div[data-baseweb="select"] [aria-haspopup="listbox"]::before,
    #agent-picker-fixed div[data-baseweb="select"] [aria-haspopup="listbox"]::after { display: none !important; }
    /* Ensure dropdown menu appears above everything */
    div[data-baseweb="popover"] { z-index: 3000 !important; }

    /* File uploader at bottom-right */
    div[data-testid="stFileUploader"] {
        position: fixed; right: 24px; bottom: 32px; z-index: 2499; /* above chat bar */
        width: var(--uploader-size) !important; height: var(--uploader-size) !important;
    }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        min-height: var(--uploader-size) !important; height: var(--uploader-size) !important; width: var(--uploader-size) !important;
        padding: 0 !important;
        border-radius: 9999px !important;
        border: 1px solid #374151 !important;
        background: #1f2937 !important; color: #e5e7eb !important;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4);
        cursor: pointer;
    }
    /* Hide default instructional text; show paperclip icon instead */
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * { display: none !important; }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]::before {
        content: "📎";
        font-size: 15px; line-height: 15px;
        color: #e5e7eb;
    }
    /* Keep just enough breathing room at the bottom so fixed input doesn't cover text */
    section[data-testid="stSidebar"] ~ div [data-testid="stVerticalBlock"]:last-child {
        padding-bottom: calc(var(--chat-height) + 40px);
    }

    /* Assistant thinking animation */
    @keyframes blink {
        0% { opacity: .2; }
        20% { opacity: 1; }
        100% { opacity: .2; }
    }
    .thinking { color: rgba(49,51,63,0.8); }
    .thinking span { animation-name: blink; animation-duration: 1.4s; animation-iteration-count: infinite; animation-fill-mode: both; }
    .thinking span:nth-child(2) { animation-delay: .2s; }
    .thinking span:nth-child(3) { animation-delay: .4s; }

    /* Message content readability */
    div[data-testid="stChatMessageContent"] {
        background: #111827 !important;
        color: #e5e7eb !important;
        border: 0 !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        font-size: var(--chat-font-size) !important; line-height: 1.75 !important;
    }
    /* Ensure all message wrappers are flat and dark — no white borders */
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] > div,
    [data-testid="stChatMessage"] *:where([style*="border" i]) {
        background: transparent !important; border: 0 !important; box-shadow: none !important; outline: none !important;
    }
    [data-testid="stChatMessage"] { padding: 6px 0 !important; }
    /* Remove blanket last-child styling to avoid oversized white blocks covering content */
    /* Headings scale with base chat font size */
    div[data-testid="stChatMessageContent"] h1 { font-size: calc(var(--chat-font-size) + 11px) !important; line-height: 1.25 !important; margin: 0.35rem 0 0.5rem !important; }
    div[data-testid="stChatMessageContent"] h2 { font-size: calc(var(--chat-font-size) + 7px) !important; line-height: 1.3 !important; margin: 0.3rem 0 0.45rem !important; }
    div[data-testid="stChatMessageContent"] h3 { font-size: calc(var(--chat-font-size) + 3px) !important; line-height: 1.35 !important; margin: 0.25rem 0 0.4rem !important; }
    div[data-testid="stChatMessageContent"] h4 { font-size: calc(var(--chat-font-size) + 1px) !important; line-height: 1.4 !important; margin: 0.25rem 0 0.35rem !important; }
    /* Lists */
    div[data-testid="stChatMessageContent"] ul { padding-left: 1.25rem !important; margin: 0.35rem 0 0.35rem 0 !important; }
    div[data-testid="stChatMessageContent"] li { margin: 0.2rem 0 !important; font-size: var(--chat-font-size) !important; }
    pre, code { background: #0f172a !important; color: #e5e7eb !important; }
    pre { border: 1px solid #374151 !important; border-radius: 8px; padding: 8px 10px; }
    a { color: #93c5fd !important; text-decoration: underline; }
    [data-testid="stFormSubmitButton"] button { background: #111827 !important; color: #e5e7eb !important; border: 1px solid #374151 !important; }

    /* Remove borders from status/toast components for a flat look */
    [data-testid*="stStatus" i], [data-testid*="toast" i], [data-testid*="Toast" i] {
        background: #0b0f12 !important; color: #e5e7eb !important; border: 0 !important; box-shadow: none !important;
    }
    [data-testid*="stStatus" i] * { color: #e5e7eb !important; }

    /* Remove any remaining white outlines from container blocks */
    [data-testid="stVerticalBlock"], [data-testid="stMainBlockContainer"], .block-container {
        border: 0 !important; box-shadow: none !important;
    }

    /* Hide any floating/help/parentheses-like circular icon near bottom-right */
    [data-testid="stStatusWidget"],
    [data-testid="stStatusWidget"] *,
    button[aria-label*="help" i],
    button[title*="help" i],
    button[aria-label*="keyboard" i],
    button[title*="keyboard" i],
    div[aria-label*="help" i],
    div[title*="help" i] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# (Relocation script moved below, after the selectbox + chat input are rendered)

# Solid white bar behind bottom controls to eliminate any residual dark line
## Remove legacy bottom bar artifacts (no-op now)
# st.markdown("""<style>#white-bottom-bar,#white-bottom-bar-2{display:none}</style>""", unsafe_allow_html=True)

# Remove legacy fallback form CSS (we now rely solely on st.chat_input)

# Render the small uploader (styled above). Label hidden to avoid text.
bottom_files = st.file_uploader(
    label="Attach files",
    type=["pdf", "txt", "md", "json", "csv"],
    accept_multiple_files=True,
    key="bottom_file_uploader",
    label_visibility="collapsed",
)

# Anchor + fixed container for agent picker relocation
st.markdown('<div id="agent-picker-anchor"></div><div id="agent-picker-fixed"></div>', unsafe_allow_html=True)

# Agent picker near the chat bar (styled to a small icon button)
AGENT_OPTIONS = [
    "Agent",
    "Communications summary",
    "Overall summary",
    "Action plan",
    "Decision: Civilians",
    "Decision: Adversaries",
]
agent_choice = st.selectbox("Agent picker", AGENT_OPTIONS, index=0, key="agent_selector", label_visibility="collapsed")

user_text = st.chat_input("Ask anything…")

# Robustly relocate the agent picker: anchor-aware and resilient to rerenders
st.markdown(
        """
        <script>
        (function(){
            function positionNextToInput(){
                try {
                    var fixed = document.getElementById('agent-picker-fixed');
                    var input = document.querySelector('[data-testid="stChatInput"]');
                    var picker = fixed ? fixed.querySelector('div[data-baseweb="select"]') : null;
                    if(!fixed || !input || !picker) return;
                    var ir = input.getBoundingClientRect();
                    var pr = picker.getBoundingClientRect();
                    var gap = 14;
                    var left = Math.max(12, Math.round(ir.left - (pr.width || 140) - gap));
                    fixed.style.left = left + 'px';
                    var vAdj = ((ir.height || 62) - (pr.height || 40))/2;
                    fixed.style.bottom = (32 + vAdj) + 'px';
                } catch(e){}
            }
            function movePicker(){
                try {
                    var anchor = document.getElementById('agent-picker-anchor');
                    var fixed = document.getElementById('agent-picker-fixed');
                    if(!anchor || !fixed) return;
                    // Find the selectbox block rendered after the anchor
                    var sib = anchor.nextElementSibling;
                    var attempts = 0, target = null;
                    while(sib && attempts < 8){
                        if (sib.querySelector) {
                            var cand = sib.querySelector('div[data-baseweb="select"]');
                            if(cand){ target = cand; break; }
                        }
                        sib = sib.nextElementSibling; attempts++;
                    }
                    if(target && !fixed.contains(target)){
                        // Hide original wrapper to prevent a ghost slot
                        var wrapper = target.closest('[data-testid="stSelectbox"]') || target.parentElement;
                        if(wrapper) wrapper.style.display = 'none';
                        fixed.appendChild(target);
                        setTimeout(positionNextToInput, 0);
                    } else {
                        positionNextToInput();
                    }
                } catch(e){}
            }
            // Initial try and observers
            setTimeout(function(){ movePicker(); }, 50);
            var mo = new MutationObserver(function(){ movePicker(); });
            mo.observe(document.body, {subtree:true, childList:true});
            window.addEventListener('resize', positionNextToInput);
        })();
        </script>
        """,
        unsafe_allow_html=True,
)
# (Old JS relocation removed; CSS now fixes position)

# If a re-run was requested, substitute the queued prompt
if user_text is None and st.session_state.get("queued_prompt"):
    user_text = st.session_state.pop("queued_prompt")
    # Tag that this is a re-run (optional future use)
    st.session_state["last_rerun"] = True
else:
    st.session_state["last_rerun"] = False

if user_text is not None:
    # Immediate, global feedback that a prompt was sent
    try:
        st.toast("Message sent — processing…", icon="💬")
    except Exception:
        st.info("Message sent — processing…")

    # Prefer relaxed composition so we don't hard-fail when documents are incomplete
    payload = _compose_input_relaxed(user_text, bottom_files)
    if not payload:
        st.warning("Please enter some text or attach a file.")
    else:
        # Map agent selection (bottom picker) to backend route key
        choice = st.session_state.get("agent_selector", "Agent")
        agent_key = None
        if choice == "Agent":
            agent_key = None  # auto-routing/classification
        elif choice == "Communications summary":
            agent_key = "communications_summary"
        elif choice == "Overall summary":
            agent_key = "overall_summary"
        elif choice == "Action plan":
            agent_key = "action_planner"
        elif choice == "Decision: Civilians":
            agent_key = "decision_civilians"
        elif choice == "Decision: Adversaries":
            agent_key = "decision_adversaries"

        # Show user message once and freeze its rendered content
        st.session_state.messages.append({"role": "user", "content": user_text, "frozen_markdown": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        st.markdown(
            """
            <script>
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
            </script>
            """,
            unsafe_allow_html=True,
        )

        # Assistant response bubble
        with st.chat_message("assistant"):
            placeholder = st.empty()
            _render_thinking_skeleton(placeholder)
            try:
                status = st.status("Processing with agent…", state="running")
            except Exception:
                status = None
            st.markdown(
                """
                <script>
                window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                </script>
                """,
                unsafe_allow_html=True,
            )

            # Compose again (ensures latest mode settings applied)
            payload = _compose_input_relaxed(user_text, bottom_files)
            if not payload:
                local = _generate_local_fallback(user_text, bottom_files)
                _stream_text_simulated(local, placeholder)
                st.session_state.messages.append({"role": "assistant", "content": local, "frozen_markdown": local, "prompt": user_text})
                if status is not None:
                    try: status.update(label="Local answer", state="complete")
                    except Exception: pass
            else:
                if not _probe_backend(_api_url(), timeout=1.0):
                    local = _generate_local_fallback(user_text, bottom_files)
                    _stream_text_simulated(local, placeholder)
                    st.session_state.messages.append({"role": "assistant", "content": local, "frozen_markdown": local, "prompt": user_text})
                    if status is not None:
                        try: status.update(label="Local answer (offline)", state="complete")
                        except Exception: pass
                else:
                    body = {
                        "input_as_text": payload,
                        "enable_web": bool(st.session_state.get("enable_web", False)),
                        "web_depth": st.session_state.get("web_depth", "advanced"),
                        "web_max_results": int(st.session_state.get("web_max_results", 6)),
                        "target_word_count": int(st.session_state.get("target_word_count", 1800)),
                        "response_time_budget_ms": int(float(st.session_state.get("response_budget", 20)) * 1000),
                        "verbose_mode": bool(st.session_state.get("verbose_mode", False)),
                    }
                    if agent_key:
                        body["target_agent"] = agent_key
                    reply, ok = _stream_agent_response(body, placeholder)
                    if not ok:
                        local = _generate_local_fallback(user_text, bottom_files)
                        _stream_text_simulated(local, placeholder)
                        reply = local
                        frozen_md = local
                    else:
                        frozen_md = _format_reply_markdown(reply or "")
                    st.session_state.messages.append({"role": "assistant", "content": reply or "", "frozen_markdown": frozen_md or "", "prompt": user_text})
                    if status is not None:
                        try: status.update(label="Reply ready", state="complete")
                        except Exception: pass
