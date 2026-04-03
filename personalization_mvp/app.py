#!/usr/bin/env python3
"""Streamlit UI — calls run_pipeline() directly, no subprocess, no Flask API."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from pipeline import PipelineConfig, run_pipeline

ROOT_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT_DIR / "runs"
STAGES = ["extract", "analyze", "generate", "diff", "judge", "retry"]


# ── session state ─────────────────────────────────────────────────────────────

def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "pipeline_status": "idle",
        "uploaded_data": None,
        "uploaded_csv_bytes": None,
        "uploaded_filename": None,
        "uploaded_fingerprint": None,
        "current_stage": None,
        "results": {},
        "upload_error": None,
        "channel": "e.g., amazon_animal_health",
        "expertise": "e.g., veterinary_expert",
        "generation_model": "gpt-4.1",
        "reasoning_model": "gpt-4.1-mini",
        "audit_model": "gpt-4.1-nano",
        "product_index": 0,
        "run_id": None,
        "error": None,
        "logs": [],
        "events_queue": None,
        "runner_thread": None,
        "run_dir": None,
        "enable_retry": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── config loaders ────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_profiles() -> tuple[list[str], list[str]]:
    payload = _load_json(ROOT_DIR / "channel_tone_specifications.json")
    channels = ["e.g., amazon_animal_health"] + sorted((payload.get("channels") or {}).keys())
    expertise = ["e.g., veterinary_expert"] + sorted((payload.get("expertise_profiles") or {}).keys())
    return channels, expertise


def load_field_specs() -> dict[str, Any]:
    return _load_json(ROOT_DIR / "field_specifications.json").get("fields", {})


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _decode_csv(raw: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode CSV with utf-8-sig or latin-1")


def parse_csv_rows(raw: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(_decode_csv(raw))))


def _normalize(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").split()).strip().lower()


def find_header_row(rows: list[list[str]]) -> int:
    for i, row in enumerate(rows[:10]):
        cols = [_normalize(c) for c in row]
        if "ean" in cols and "marque" in cols and "designation" in cols:
            return i
    raise ValueError("Could not detect CSV header row")


def find_status_row(rows: list[list[str]], start: int) -> int | None:
    markers = {"TODO", "OK", "NON", "OUI"}
    for i in range(start, len(rows)):
        values = {c.strip().upper() for c in rows[i] if c.strip()}
        if "TODO" in values and values & markers:
            return i
    return None


def get_data_row_indices(rows: list[list[str]]) -> list[int]:
    header_idx = find_header_row(rows)
    status_idx = find_status_row(rows, header_idx + 1)
    end = status_idx if status_idx is not None else len(rows)
    return [idx for idx in range(header_idx + 1, end) if any(c.strip() for c in rows[idx])]


def parse_uploaded_to_df(raw: bytes) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), header=None, dtype=str, keep_default_na=False, encoding=enc)
        except Exception as exc:
            last_err = exc
    raise ValueError(f"Unable to parse uploaded CSV: {last_err}")


# ── pipeline worker ───────────────────────────────────────────────────────────

def _pipeline_worker(events: queue.Queue, cfg: PipelineConfig) -> None:
    try:
        run_pipeline(cfg, on_event=lambda e: events.put(e))
    except Exception as exc:
        events.put({"type": "error", "message": str(exc)})


def _push_log(message: str) -> None:
    st.session_state["logs"].append(message)
    st.session_state["logs"] = st.session_state["logs"][-300:]


def poll_events() -> None:
    q = st.session_state.get("events_queue")
    if not q:
        return
    while True:
        try:
            evt = q.get_nowait()
        except queue.Empty:
            break
        typ = evt.get("type")
        if typ == "stage":
            st.session_state["current_stage"] = str(evt.get("stage", ""))
        elif typ == "log":
            _push_log(str(evt.get("message", "")))
        elif typ == "complete":
            artifacts = evt.get("artifacts", {})
            loaded: dict[str, Any] = {}
            for name, path in artifacts.items():
                p = Path(str(path))
                if p.exists():
                    loaded[name] = _load_json(p) if p.suffix == ".json" else p.read_text(encoding="utf-8", errors="ignore")
            st.session_state["results"] = loaded
            st.session_state["pipeline_status"] = "complete"
            st.session_state["current_stage"] = "retry"
            st.session_state["error"] = None
            st.session_state["events_queue"] = None
            st.session_state["runner_thread"] = None
            _push_log("Pipeline complete.")
        elif typ == "error":
            st.session_state["pipeline_status"] = "failed"
            st.session_state["error"] = str(evt.get("message", "Unknown error"))
            st.session_state["events_queue"] = None
            st.session_state["runner_thread"] = None
            _push_log(f"ERROR: {st.session_state['error']}")


def start_run() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        st.session_state["pipeline_status"] = "failed"
        st.session_state["error"] = "OPENAI_API_KEY is not set in environment."
        return

    run_id = datetime.now().strftime("local_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    file_name = st.session_state.get("uploaded_filename") or "uploaded_matrix.csv"
    csv_path = run_dir / file_name
    csv_path.write_bytes(st.session_state["uploaded_csv_bytes"])

    cfg = PipelineConfig(
        csv_path=csv_path,
        index=int(st.session_state["product_index"]),
        channel=st.session_state["channel"],
        expertise=st.session_state["expertise"],
        generation_model=st.session_state.get("generation_model", "gpt-4.1"),
        reasoning_model=st.session_state.get("reasoning_model", "gpt-4.1-mini"),
        audit_model=st.session_state.get("audit_model", "gpt-4.1-nano"),
        out_dir=run_dir,
        enable_retry=st.session_state.get("enable_retry", True),
        field_specs_path=ROOT_DIR / "field_specifications.json",
        channel_tone_path=ROOT_DIR / "channel_tone_specifications.json",
        descriptions_specs_path=ROOT_DIR / "descriptions_content" / "description_specs.json",
        seo_specs_path=ROOT_DIR / "seo_fields" / "seo_fields_specs.json",
        analyze_specs_path=ROOT_DIR / "analyze_specs.json",
    )

    q: queue.Queue = queue.Queue()
    thread = threading.Thread(target=_pipeline_worker, args=(q, cfg), daemon=True)
    thread.start()

    st.session_state.update({
        "run_id": run_id,
        "run_dir": str(run_dir),
        "events_queue": q,
        "runner_thread": thread,
        "pipeline_status": "processing",
        "current_stage": "extract",
        "error": None,
        "results": {},
        "upload_error": None,
        "logs": ["Run started."],
    })


# ── result helpers ────────────────────────────────────────────────────────────

def get_retry_map(results: dict[str, Any]) -> dict[str, Any]:
    retry = results.get("retry", {})
    if isinstance(retry, dict) and isinstance(retry.get("generated_fields"), dict):
        return retry["generated_fields"]
    return {}


def get_judge_payload(results: dict[str, Any]) -> dict[str, Any]:
    judge = results.get("judge", {})
    return judge if isinstance(judge, dict) else {}


def get_source_meta(results: dict[str, Any]) -> dict[str, Any]:
    inp = results.get("input", {})
    if isinstance(inp, dict) and isinstance(inp.get("source"), dict):
        return inp["source"]
    return {}


# ── comparison table ──────────────────────────────────────────────────────────

def _original_values_from_input(results: dict[str, Any]) -> dict[str, str]:
    inp = results.get("input", {})
    flat: dict[str, str] = {}
    for section in ("commercial_seo_inputs", "content_body_inputs", "core_identity", "scientific_regulatory_inputs"):
        block = inp.get(section) or {}
        if isinstance(block, dict):
            flat.update({k: str(v) for k, v in block.items() if v is not None})
    return flat


_FIELD_ALIASES: dict[str, str] = {
    "Titre court du produit (60 caractères max)": "Titre court",
    "Description courte (...) 175 caractères": "Description courte (175 caractères)",
}


def _health(field: str, text: str, field_specs: dict[str, Any]) -> str:
    spec = field_specs.get(field, {})
    n = len(text)
    min_c, max_c, target = spec.get("min_chars"), spec.get("max_chars"), spec.get("target_chars")
    if min_c is not None and n < int(min_c):
        return "(Under)"
    if max_c is not None and n > int(max_c):
        return "(Over)"
    if target is not None:
        return "(OK)" if abs(n - int(target)) <= 10 else ("(Under)" if n < int(target) else "(Over)")
    return "(OK)"


def build_comparison_df(field_specs: dict[str, Any]) -> pd.DataFrame:
    results = st.session_state.get("results", {})
    retry_map = get_retry_map(results)
    originals = _original_values_from_input(results)

    def get_original(field: str) -> str:
        val = originals.get(field)
        if val:
            return val
        alias = _FIELD_ALIASES.get(field)
        return originals.get(alias, "") if alias else ""

    records = []
    for field in field_specs:
        original = get_original(field)
        optimized = str(retry_map.get(field, original) or "")
        records.append({
            "Field Name": field,
            "Original Value (from CSV)": original,
            "Final Optimized Value": optimized,
            "Health Check": _health(field, optimized, field_specs),
            "Generated Characters": len(optimized),
        })
    return pd.DataFrame(records)


def style_comparison_df(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def color_row(row: pd.Series) -> list[str]:
        v = row["Health Check"]
        style = {
            "(OK)": "background-color: #E8F5EE; color: #1F6B4F;",
            "(Under)": "background-color: #FFF6E5; color: #8A5A00;",
            "(Over)": "background-color: #FDECEC; color: #8B1F1F;",
        }.get(v, "")
        return [style if col == "Health Check" else "" for col in df.columns]
    return df.style.apply(color_row, axis=1)


def build_download_csv_bytes(field_specs: dict[str, Any]) -> bytes:
    df = st.session_state["uploaded_data"].copy()
    raw_rows = parse_csv_rows(st.session_state["uploaded_csv_bytes"])
    data_rows = get_data_row_indices(raw_rows)

    source = get_source_meta(st.session_state.get("results", {}))
    product_index = int(source.get("product_index_0_based", st.session_state.get("product_index", 0)))
    if product_index < 0 or product_index >= len(data_rows):
        raise IndexError("Product index out of range")

    absolute_row = data_rows[product_index]
    retry_map = get_retry_map(st.session_state.get("results", {}))

    max_col = max(int(spec.get("doc_column_index_0_based", 0)) for spec in field_specs.values())
    needed_cols = max(max_col + 1, df.shape[1])
    if df.shape[1] < needed_cols:
        df = df.reindex(columns=range(needed_cols), fill_value="")
    if df.shape[0] <= absolute_row:
        df = df.reindex(index=range(absolute_row + 1), fill_value="")

    for field, value in retry_map.items():
        spec = field_specs.get(field)
        if not spec:
            continue
        col_idx = int(spec.get("doc_column_index_0_based", -1))
        if col_idx >= 0:
            df.iat[absolute_row, col_idx] = "" if value is None else str(value)

    out = io.StringIO()
    df.to_csv(out, index=False, header=False)
    return out.getvalue().encode("utf-8-sig")


# ── rendering ─────────────────────────────────────────────────────────────────

def render_timeline(current_stage: str | None, status: str) -> None:
    idx_map = {s: i for i, s in enumerate(STAGES)}
    current_idx = idx_map.get(current_stage or "", 0)
    cols = st.columns(len(STAGES))
    for i, stage in enumerate(STAGES):
        if status == "complete":
            state = "done"
        elif status == "processing":
            state = "done" if i < current_idx else ("active" if i == current_idx else "inactive")
        else:
            state = "inactive"

        if state == "done":
            circle = '<div class="node node-done">✓</div>'
            label = f"<div class='node-label'>{stage}</div>"
        elif state == "active":
            circle = '<div class="node node-active"></div>'
            label = f"<div class='node-label active'>{stage}</div>"
        else:
            circle = '<div class="node node-inactive"></div>'
            label = f"<div class='node-label'>{stage}</div>"
        cols[i].markdown(f"<div class='timeline-cell'>{circle}{label}</div>", unsafe_allow_html=True)


def render_idle() -> None:
    with st.container():
        st.markdown('<div class="panel idle-panel">', unsafe_allow_html=True)
        st.markdown("<div class='idle-icon'>📄</div>", unsafe_allow_html=True)
        st.markdown("### Upload your product matrix to begin personalization")
        st.markdown("Select your parameters in the sidebar and drag your CSV file here.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_processing() -> None:
    with st.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Pipeline Execution")
        render_timeline(st.session_state.get("current_stage"), "processing")
        st.markdown("#### Live Logs")
        logs_text = "\n".join(st.session_state.get("logs", [])[-80:])
        st.code(logs_text if logs_text else "Waiting for logs...", language="bash")
        st.markdown("</div>", unsafe_allow_html=True)


def render_complete(field_specs: dict[str, Any]) -> None:
    df = build_comparison_df(field_specs)
    styled = style_comparison_df(df)
    judge_payload = get_judge_payload(st.session_state.get("results", {}))
    with st.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Review & Export")
        render_timeline("retry", "complete")
        st.caption("Pipeline Complete")
        st.dataframe(styled, use_container_width=True, hide_index=True)
        with st.expander("The Judge's Verdict", expanded=True):
            st.json(judge_payload)
        st.markdown("</div>", unsafe_allow_html=True)


def render_failed() -> None:
    with st.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.error(st.session_state.get("error") or "Pipeline failed.")
        logs_text = "\n".join(st.session_state.get("logs", [])[-80:])
        st.code(logs_text if logs_text else "No logs available.", language="bash")
        st.markdown("</div>", unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown("""
        <style>
        .block-container { padding: 3rem 3rem 2rem 3rem; background: #FFFFFF; }
        [data-testid="stSidebar"] { background: #F8F9FA; padding-right: 1rem; border-right: 1px solid #EAECEF; }
        [data-testid="stExpander"] { border-radius: 12px; box-shadow: 0 2px 12px rgba(15,23,42,0.06); }
        .panel { border-radius: 12px; background: #FFFFFF; border: 1px solid #EEF1F4; box-shadow: 0 4px 18px rgba(15,23,42,0.06); padding: 1.5rem; margin-bottom: 1rem; }
        .idle-panel { text-align: center; padding: 3rem; color: #2C3E50; }
        .idle-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
        .timeline-cell { display: flex; flex-direction: column; align-items: center; gap: 0.35rem; margin-bottom: 0.5rem; }
        .node { width: 24px; height: 24px; border-radius: 999px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700; }
        .node-done { background: #2D9CDB; color: white; }
        .node-active { border: 3px solid #2D9CDB; background: #EAF6FD; }
        .node-inactive { border: 2px solid #D7DEE5; background: #F4F7FA; }
        .node-label { text-transform: uppercase; letter-spacing: 0.02em; font-size: 11px; color: #8A95A3; }
        .node-label.active { color: #2C3E50; font-weight: 700; }
        .stButton > button, [data-testid="baseButton-primary"] { background-color: #2D9CDB !important; color: white !important; border: none !important; border-radius: 10px !important; }
        </style>
    """, unsafe_allow_html=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Product Personalization AI - MVP", layout="wide")
    initialize_state()
    inject_css()

    field_specs = load_field_specs()
    channels, expertise_profiles = load_profiles()

    if st.session_state.get("pipeline_status") == "processing":
        poll_events()

    st.title("Product Personalization AI - MVP")

    with st.sidebar:
        st.header("Pipeline Setup")

        uploaded_file = st.file_uploader("Upload Product Matrix (CSV)", type="csv")
        if uploaded_file is not None:
            raw = uploaded_file.getvalue()
            fingerprint = hashlib.sha256(raw).hexdigest()
            if st.session_state.get("uploaded_fingerprint") != fingerprint:
                try:
                    parsed_df = parse_uploaded_to_df(raw)
                    parse_csv_rows(raw)
                except Exception as exc:
                    st.session_state.update({
                        "uploaded_data": None,
                        "uploaded_csv_bytes": None,
                        "uploaded_filename": uploaded_file.name,
                        "uploaded_fingerprint": fingerprint,
                        "upload_error": f"Could not parse uploaded CSV: {exc}",
                        "pipeline_status": "failed",
                        "error": f"Could not parse uploaded CSV: {exc}",
                    })
                else:
                    st.session_state.update({
                        "uploaded_csv_bytes": raw,
                        "uploaded_filename": uploaded_file.name,
                        "uploaded_fingerprint": fingerprint,
                        "uploaded_data": parsed_df,
                        "upload_error": None,
                        "pipeline_status": "idle",
                        "results": {},
                        "error": None,
                        "product_index": 0,
                    })

        if st.session_state.get("uploaded_data") is not None:
            try:
                data_rows = get_data_row_indices(parse_csv_rows(st.session_state["uploaded_csv_bytes"]))
                structure_error = None
            except Exception as exc:
                data_rows = []
                structure_error = f"Could not locate CSV product rows: {exc}"
            max_idx = max(0, len(data_rows) - 1)

            st.number_input("Product Index", min_value=0, max_value=max_idx, key="product_index", help=f"Range: 0-{max_idx}")
            st.selectbox("Target Channel", options=channels, key="channel")
            st.selectbox("Expertise Profile", options=expertise_profiles, key="expertise")
            st.text_input("Generation Model", key="generation_model", help="Descriptions + retry (e.g. gpt-4.1)")
            st.text_input("Reasoning Model", key="reasoning_model", help="SEO + judge (e.g. gpt-4.1-mini)")
            st.text_input("Audit Model", key="audit_model", help="Consistency check (e.g. gpt-4.1-nano)")
            st.checkbox("Enable Retry", key="enable_retry", help="Apply judge feedback to rewrite changed fields")

            processing = st.session_state.get("pipeline_status") == "processing"
            st.button(
                "Run Pipeline",
                type="primary",
                on_click=start_run,
                disabled=processing or structure_error is not None,
                use_container_width=True,
            )
            if structure_error:
                st.error(structure_error)

            if st.session_state.get("pipeline_status") == "complete":
                try:
                    out_bytes = build_download_csv_bytes(field_specs)
                    original_name = st.session_state.get("uploaded_filename") or "matrix.csv"
                    new_name = original_name.rsplit(".", 1)[0] + "_personalisee.csv"
                    st.download_button("Download Updated CSV", data=out_bytes, file_name=new_name, mime="text/csv", use_container_width=True)
                except Exception as exc:
                    st.error(f"Download build failed: {exc}")

                diff_data = st.session_state.get("results", {}).get("diff")
                if diff_data is not None:
                    diff_bytes = json.dumps(diff_data, ensure_ascii=False, indent=2).encode("utf-8")
                    original_name = st.session_state.get("uploaded_filename") or "matrix.csv"
                    diff_name = original_name.rsplit(".", 1)[0] + "_diff.json"
                    st.download_button("Download Diff JSON", data=diff_bytes, file_name=diff_name, mime="application/json", use_container_width=True)

    status = st.session_state.get("pipeline_status")
    if status == "idle":
        render_idle()
    elif status == "processing":
        render_processing()
        time.sleep(2)
        st.rerun()
    elif status == "complete":
        render_complete(field_specs)
    elif status == "failed":
        render_failed()


if __name__ == "__main__":
    main()
