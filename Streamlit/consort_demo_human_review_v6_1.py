"""
CONSORT Evidence Extraction Demo — Human-in-the-Loop Review V6.1
============================================================

Conference prototype supporting two inference paths:
1) PRECOMPUTED pre-GEPA test-set outputs (instant, no model call); and
2) LIVE .txt uploads processed through the v26-style router + section extractors.

The live path is implemented in ``consort_live_pipeline_v4.py`` and deliberately
stops before the verifier/adjudication/recovery stages.

Primary workflow
----------------
1. Load ``pre_gepa_test_demo_records.json`` (expected next to this .py file).
2. Select a held-out test article from a list.
3. Review CONSORT item evidence sentence-by-sentence.
4. Mark each model-extracted sentence Correct / Incorrect.
5. Optionally enable "Add missed evidence" and click exact source sentences in
   the article to add false-negative (FN) evidence for the selected item.
6. Save the human review. The app writes:
      human_feedback/human_feedback_store.json
      human_feedback/gepa_feedback.jsonl
      human_feedback/human_corrected_extractions.jsonl

Identifier contract
-------------------
The app NEVER generates, parses, or normalizes article IDs from text.
It copies identifiers directly from the precomputed export:
- sid  = exact canonical sentence ID
- pnum = model-facing within-article paragraph index
- pid  = original paragraph ID

This makes the saved human feedback directly traceable to the pre-GEPA run and
suitable for later human-in-the-loop / GEPA experiments.

Run
---
    streamlit run consort_demo_human_review_v6.py

Required:
    pip install streamlit

Live v4 supports TXT only. PDF/XML ingestion is intentionally out of scope.
"""

from __future__ import annotations

import json
import re
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import streamlit as st

try:
    from consort_live_pipeline_v4 import (
        PipelineConfig,
        preprocess_txt,
        run_txt_pipeline,
        check_asset_paths,
    )
    LIVE_PIPELINE_IMPORT_ERROR = None
except Exception as _live_exc:
    PipelineConfig = None
    preprocess_txt = None
    run_txt_pipeline = None
    check_asset_paths = None
    LIVE_PIPELINE_IMPORT_ERROR = repr(_live_exc)

try:
    from consort_synthesis_v6 import (
        bootstrap_precomputed_records,
        upsert_study_record,
        load_repository_records,
        load_all_profiles,
        profile_cache_state,
        ensure_study_profile,
        load_family_store,
        assign_profile_questions,
        resolve_pending_assignment,
        generate_family_summary,
        topic_table_rows,
        repository_paths,
    )
    SYNTHESIS_IMPORT_ERROR = None
except Exception as _synth_exc:
    bootstrap_precomputed_records = None
    upsert_study_record = None
    load_repository_records = None
    load_all_profiles = None
    profile_cache_state = None
    ensure_study_profile = None
    load_family_store = None
    assign_profile_questions = None
    resolve_pending_assignment = None
    generate_family_summary = None
    topic_table_rows = None
    repository_paths = None
    SYNTHESIS_IMPORT_ERROR = repr(_synth_exc)


# =============================================================================
# Page setup
# =============================================================================
st.set_page_config(
    page_title="CONSORT Evidence Demo",
    page_icon="📋",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR / "pre_gepa_test_demo_records.json"
FEEDBACK_DIR = APP_DIR / "human_feedback"
FEEDBACK_STORE_PATH = FEEDBACK_DIR / "human_feedback_store.json"
GEPA_FEEDBACK_PATH = FEEDBACK_DIR / "gepa_feedback.jsonl"
CORRECTED_EXTRACTIONS_PATH = FEEDBACK_DIR / "human_corrected_extractions.jsonl"
LIVE_RUN_DIR = APP_DIR / "live_runs"
PIPELINE_ASSET_DIR = APP_DIR / "pipeline_assets"
DEFAULT_PROTOTYPE_PATH = PIPELINE_ASSET_DIR / "consort_prototype_pass5_boundary_hardened_v1.json"
DEFAULT_GUIDELINE_PATH = PIPELINE_ASSET_DIR / "annotation_guideline_v3_structured.json"
DEFAULT_ROUTER_PATH = PIPELINE_ASSET_DIR / "optimized_section_labeler_multilabel.json"

# DEVELOPMENT-ONLY credential fallback. Leave blank for normal/local sharing.
# If you temporarily put your own key here, remove it before deploying the app.
DEVELOPER_OPENAI_API_KEY = ""
DEFAULT_SYNTHESIS_MODEL = "openai/gpt-5.6-terra"

SCHEMA_VERSION = "consort_hil_v2"
APP_VERSION = "CONSORT Demo v6.1"


# =============================================================================
# CONSORT UI metadata
# Keep IDs aligned with the pipeline export. Descriptions are UI-only.
# =============================================================================
CONSORT_ITEMS: List[Dict[str, str]] = [
    {"id": "2a", "label": "Scientific background and explanation of rationale", "section": "Introduction"},
    {"id": "2b", "label": "Specific objectives or hypotheses", "section": "Introduction"},
    {"id": "3a", "label": "Description of trial design", "section": "Methods"},
    {"id": "3b", "label": "Important changes to methods after trial commencement", "section": "Methods"},
    {"id": "4a", "label": "Eligibility criteria for participants", "section": "Methods"},
    {"id": "4b", "label": "Settings and locations where data were collected", "section": "Methods"},
    {"id": "5", "label": "Interventions for each group with sufficient details", "section": "Methods"},
    {"id": "6a", "label": "Completely defined pre-specified primary and secondary outcomes", "section": "Methods"},
    {"id": "6b", "label": "Changes to trial outcomes after commencement", "section": "Methods"},
    {"id": "7a", "label": "How sample size was determined", "section": "Methods"},
    {"id": "7b", "label": "Interim analyses and stopping guidelines when applicable", "section": "Methods"},
    {"id": "8a", "label": "Method used to generate the random allocation sequence", "section": "Methods"},
    {"id": "8b", "label": "Type of randomization and restriction details", "section": "Methods"},
    {"id": "9", "label": "Allocation concealment mechanism", "section": "Methods"},
    {"id": "10", "label": "Who generated, enrolled, and assigned participants", "section": "Methods"},
    {"id": "11a", "label": "Who was blinded after assignment and how", "section": "Methods"},
    {"id": "11b", "label": "Similarity of interventions if relevant", "section": "Methods"},
    {"id": "12a", "label": "Statistical methods used to compare groups", "section": "Methods"},
    {"id": "12b", "label": "Methods for additional analyses", "section": "Methods"},
    {"id": "13a", "label": "Participant flow through each trial group", "section": "Results"},
    {"id": "13b", "label": "Losses and exclusions after randomization with reasons", "section": "Results"},
    {"id": "14a", "label": "Dates defining recruitment and follow-up periods", "section": "Results"},
    {"id": "14b", "label": "Why the trial ended or was stopped", "section": "Results"},
    {"id": "15", "label": "Baseline demographic and clinical characteristics", "section": "Results"},
    {"id": "16", "label": "Numbers analyzed for each group", "section": "Results"},
    {"id": "17a", "label": "Outcome results with effect size and precision", "section": "Results"},
    {"id": "17b", "label": "Absolute and relative effect sizes for binary outcomes", "section": "Results"},
    {"id": "18", "label": "Results of ancillary analyses", "section": "Results"},
    {"id": "19", "label": "Important harms or unintended effects", "section": "Results"},
    {"id": "20", "label": "Trial limitations", "section": "Discussion"},
    {"id": "21", "label": "Generalisability of trial findings", "section": "Discussion"},
    {"id": "22", "label": "Interpretation consistent with results and evidence", "section": "Discussion"},
    {"id": "23", "label": "Registration number and registry name", "section": "Other"},
    {"id": "24", "label": "Where the full trial protocol can be accessed", "section": "Other"},
    {"id": "25", "label": "Sources of funding/support and role of funders", "section": "Other"},
]
ITEM_BY_ID = {row["id"]: row for row in CONSORT_ITEMS}
ITEM_ORDER = [row["id"] for row in CONSORT_ITEMS]


# =============================================================================
# Data loading / validation
# =============================================================================
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_bytes(raw: bytes) -> List[Dict[str, Any]]:
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, list):
        raise ValueError("Expected a JSON array of article records.")
    return obj


@st.cache_data(show_spinner=False)
def load_records_from_path(path_str: str) -> List[Dict[str, Any]]:
    with open(path_str, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("Expected a JSON array of article records.")
    return obj


def validate_article_record(record: Dict[str, Any]) -> List[str]:
    """Validate only the identifier invariants needed by the review UI."""
    warnings: List[str] = []
    article_id = str(record.get("article_id", ""))
    sentences = record.get("sentences", []) or []

    sid_map: Dict[str, Tuple[Any, Any, str]] = {}
    for s in sentences:
        sid = str(s.get("sid", ""))
        if not sid:
            warnings.append(f"{article_id}: sentence missing sid")
            continue
        signature = (s.get("pnum"), str(s.get("pid", "")), str(s.get("text", "")))
        if sid in sid_map and sid_map[sid] != signature:
            warnings.append(f"{article_id}: SID {sid} has conflicting sentence metadata")
        sid_map[sid] = signature

    for e in record.get("final_extractions", []) or []:
        sid = str(e.get("sid", ""))
        if sid not in sid_map:
            warnings.append(f"{article_id}: extracted SID {sid} is absent from canonical sentences")
            continue
        pnum, pid, text = sid_map[sid]
        if e.get("pnum") != pnum:
            warnings.append(f"{article_id}: extracted SID {sid} pnum mismatch")
        if str(e.get("pid", "")) != pid:
            warnings.append(f"{article_id}: extracted SID {sid} pid mismatch")
        if str(e.get("text", "")) != text:
            warnings.append(f"{article_id}: extracted SID {sid} text mismatch")

    return warnings


def article_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["article_id"]): r for r in records}


def sentence_index(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(s["sid"]): s for s in record.get("sentences", []) or []}


def checklist_index(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(c["item"]): c for c in record.get("checklist", []) or []}


def baseline_evidence_by_item(record: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {item: [] for item in ITEM_ORDER}
    for e in record.get("final_extractions", []) or []:
        item = str(e.get("item", ""))
        out.setdefault(item, []).append(e)
    return out


def infer_title(record: Dict[str, Any]) -> str:
    """Best-effort display title only; never used as an identifier."""
    paragraphs = record.get("paragraphs", []) or []
    for p in paragraphs[:8]:
        text = str(p.get("text", "")).strip()
        if not text or text.lower() in {"title", "abstract"}:
            continue
        if len(text) >= 20:
            return text[:140] + ("…" if len(text) > 140 else "")
    return str(record.get("article_id", "Article"))


# =============================================================================
# Feedback persistence
# =============================================================================
def empty_feedback_store() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "reviews": {},
    }


def load_feedback_store() -> Dict[str, Any]:
    if not FEEDBACK_STORE_PATH.exists():
        return empty_feedback_store()
    try:
        with open(FEEDBACK_STORE_PATH, "r", encoding="utf-8") as f:
            store = json.load(f)
        if not isinstance(store, dict) or "reviews" not in store:
            return empty_feedback_store()
        return store
    except Exception:
        return empty_feedback_store()


def review_id(article_id: str, item_id: str) -> str:
    return f"{article_id}::{item_id}"


def get_saved_review(article_id: str, item_id: str) -> Optional[Dict[str, Any]]:
    store = st.session_state.feedback_store
    return store.get("reviews", {}).get(review_id(article_id, item_id))


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def gepa_rows_from_store(store: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten only explicit human labels. Unreviewed predictions are omitted."""
    rows: List[Dict[str, Any]] = []
    for review in store.get("reviews", {}).values():
        common = {
            "schema_version": SCHEMA_VERSION,
            "article_id": review.get("article_id"),
            "item_id": review.get("item_id"),
            "pipeline_stage": review.get("pipeline_stage"),
            "review_complete": review.get("review_complete", False),
            "reviewed_at": review.get("reviewed_at"),
        }

        for row in review.get("baseline_evidence", []) or []:
            decision = row.get("human_decision")
            if decision not in {"correct", "incorrect"}:
                continue
            human_label = 1 if decision == "correct" else 0
            rows.append(
                {
                    **common,
                    "sid": row.get("sid"),
                    "pnum": row.get("pnum"),
                    "pid": row.get("pid"),
                    "text": row.get("text"),
                    "model_label": 1,
                    "human_label": human_label,
                    "error_type": "TP" if human_label == 1 else "FP",
                    "source": "model_prediction_review",
                    "human_comment": row.get("human_comment", ""),
                }
            )

        for row in review.get("added_evidence", []) or []:
            rows.append(
                {
                    **common,
                    "sid": row.get("sid"),
                    "pnum": row.get("pnum"),
                    "pid": row.get("pid"),
                    "text": row.get("text"),
                    "model_label": 0,
                    "human_label": 1,
                    "error_type": "FN",
                    "source": "human_added_evidence",
                    "human_comment": row.get("human_comment", ""),
                }
            )

    return rows


def corrected_rows_from_store(store: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for review in store.get("reviews", {}).values():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "article_id": review.get("article_id"),
                "item_id": review.get("item_id"),
                "pipeline_stage": review.get("pipeline_stage"),
                "baseline_status": review.get("baseline_status"),
                "fn_review_complete": review.get("fn_review_complete", False),
                "review_complete": review.get("review_complete", False),
                "human_evidence_status": review.get("human_evidence_status"),
                "confirmed_correct_sids": review.get("confirmed_correct_sids", []),
                "false_positive_sids": review.get("false_positive_sids", []),
                "false_negative_added_sids": review.get("false_negative_added_sids", []),
                "unreviewed_baseline_sids": review.get("unreviewed_baseline_sids", []),
                "effective_evidence_sids": review.get("effective_evidence_sids", []),
                "human_confirmed_evidence_sids": review.get("human_confirmed_evidence_sids", []),
                "item_comment": review.get("item_comment", ""),
                "reviewed_at": review.get("reviewed_at"),
            }
        )
    return rows


def persist_feedback_store(store: Dict[str, Any]) -> None:
    store["updated_at"] = utc_now()
    atomic_write_json(FEEDBACK_STORE_PATH, store)
    write_jsonl(GEPA_FEEDBACK_PATH, gepa_rows_from_store(store))
    write_jsonl(CORRECTED_EXTRACTIONS_PATH, corrected_rows_from_store(store))


# =============================================================================
# Widget state helpers
# =============================================================================
def state_prefix(article_id: str, item_id: str) -> str:
    safe_a = re.sub(r"[^A-Za-z0-9_-]+", "_", article_id)
    safe_i = re.sub(r"[^A-Za-z0-9_-]+", "_", item_id)
    return f"{safe_a}__{safe_i}"


def decision_key(article_id: str, item_id: str, sid: str) -> str:
    return f"decision__{state_prefix(article_id, item_id)}__{sid}"


def comment_key(article_id: str, item_id: str, sid: str) -> str:
    return f"comment__{state_prefix(article_id, item_id)}__{sid}"


def fn_comment_key(article_id: str, item_id: str, sid: str) -> str:
    return f"fn_comment__{state_prefix(article_id, item_id)}__{sid}"


def added_sids_key(article_id: str, item_id: str) -> str:
    return f"added_sids__{state_prefix(article_id, item_id)}"


def fn_complete_key(article_id: str, item_id: str) -> str:
    return f"fn_complete__{state_prefix(article_id, item_id)}"


def item_comment_key(article_id: str, item_id: str) -> str:
    return f"item_comment__{state_prefix(article_id, item_id)}"


def add_mode_key(article_id: str, item_id: str) -> str:
    return f"add_mode__{state_prefix(article_id, item_id)}"


def focused_sid_key(article_id: str) -> str:
    return f"focused_sid__{re.sub(r'[^A-Za-z0-9_-]+', '_', article_id)}"


def initialise_review_widget_state(
    article_id: str,
    item_id: str,
    baseline_rows: List[Dict[str, Any]],
) -> None:
    saved = get_saved_review(article_id, item_id) or {}
    saved_baseline = {str(x.get("sid")): x for x in saved.get("baseline_evidence", []) or []}

    for row in baseline_rows:
        sid = str(row["sid"])
        dkey = decision_key(article_id, item_id, sid)
        ckey = comment_key(article_id, item_id, sid)
        if dkey not in st.session_state:
            saved_decision = saved_baseline.get(sid, {}).get("human_decision", "unreviewed")
            st.session_state[dkey] = saved_decision
        if ckey not in st.session_state:
            st.session_state[ckey] = saved_baseline.get(sid, {}).get("human_comment", "")

    akey = added_sids_key(article_id, item_id)
    if akey not in st.session_state:
        st.session_state[akey] = [str(x["sid"]) for x in saved.get("added_evidence", []) or []]

    for row in saved.get("added_evidence", []) or []:
        sid = str(row["sid"])
        fkey = fn_comment_key(article_id, item_id, sid)
        if fkey not in st.session_state:
            st.session_state[fkey] = row.get("human_comment", "")

    fcomplete = fn_complete_key(article_id, item_id)
    if fcomplete not in st.session_state:
        st.session_state[fcomplete] = bool(saved.get("fn_review_complete", False))

    ikey = item_comment_key(article_id, item_id)
    if ikey not in st.session_state:
        st.session_state[ikey] = saved.get("item_comment", "")

    mkey = add_mode_key(article_id, item_id)
    if mkey not in st.session_state:
        st.session_state[mkey] = False


# =============================================================================
# Review construction
# =============================================================================
def build_review_payload(
    record: Dict[str, Any],
    item_id: str,
    baseline_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    article_id = str(record["article_id"])
    s_index = sentence_index(record)

    reviewed_baseline: List[Dict[str, Any]] = []
    confirmed_correct_sids: List[str] = []
    false_positive_sids: List[str] = []
    unreviewed_sids: List[str] = []

    for row in baseline_rows:
        sid = str(row["sid"])
        decision = st.session_state.get(decision_key(article_id, item_id, sid), "unreviewed")
        comment = st.session_state.get(comment_key(article_id, item_id, sid), "")
        if decision == "correct":
            confirmed_correct_sids.append(sid)
        elif decision == "incorrect":
            false_positive_sids.append(sid)
        else:
            unreviewed_sids.append(sid)

        reviewed_baseline.append(
            {
                "sid": sid,
                "pnum": row.get("pnum"),
                "pid": str(row.get("pid", "")),
                "text": row.get("text", ""),
                "source_extractors": row.get("source_extractors", []),
                "verification_status": row.get("verification_status"),
                "human_decision": decision,
                "human_comment": comment,
            }
        )

    baseline_sid_set = {str(x["sid"]) for x in baseline_rows}
    added_sids = [
        sid
        for sid in st.session_state.get(added_sids_key(article_id, item_id), [])
        if sid in s_index and sid not in baseline_sid_set
    ]

    added_evidence: List[Dict[str, Any]] = []
    for sid in added_sids:
        sentence = s_index[sid]
        added_evidence.append(
            {
                "sid": sid,
                "pnum": sentence.get("pnum"),
                "pid": str(sentence.get("pid", "")),
                "text": sentence.get("text", ""),
                "source": "human_fn_addition",
                "human_comment": st.session_state.get(fn_comment_key(article_id, item_id, sid), ""),
            }
        )

    fn_review_complete = bool(st.session_state.get(fn_complete_key(article_id, item_id), False))
    review_complete = (len(unreviewed_sids) == 0) and fn_review_complete

    # Effective corrected set preserves unreviewed baseline predictions unless
    # explicitly rejected, then adds human FN selections. This is useful for a
    # provisional corrected snapshot, but GEPA training uses ONLY explicit
    # human labels (see gepa_rows_from_store).
    effective_baseline = [
        str(row["sid"])
        for row in baseline_rows
        if str(row["sid"]) not in set(false_positive_sids)
    ]
    effective_evidence_sids = list(dict.fromkeys(effective_baseline + added_sids))
    human_confirmed_evidence_sids = list(dict.fromkeys(confirmed_correct_sids + added_sids))

    if review_complete:
        human_evidence_status = "reported" if effective_evidence_sids else "no_evidence_found"
    elif human_confirmed_evidence_sids:
        human_evidence_status = "partially_reviewed_with_evidence"
    else:
        human_evidence_status = "partially_reviewed"

    checklist = checklist_index(record).get(item_id, {})

    return {
        "schema_version": SCHEMA_VERSION,
        "review_id": review_id(article_id, item_id),
        "article_id": article_id,
        "item_id": item_id,
        "item_label": ITEM_BY_ID.get(item_id, {}).get("label", ""),
        "pipeline_stage": record.get("pipeline_stage", "pre_gepa_base_model"),
        "split": record.get("split"),
        "baseline_status": checklist.get("status"),
        "active_in_pipeline": checklist.get("active_in_pipeline"),
        "excluded_from_test_metric": checklist.get("excluded_from_test_metric"),
        "baseline_evidence": reviewed_baseline,
        "added_evidence": added_evidence,
        "confirmed_correct_sids": confirmed_correct_sids,
        "false_positive_sids": false_positive_sids,
        "false_negative_added_sids": added_sids,
        "unreviewed_baseline_sids": unreviewed_sids,
        "effective_evidence_sids": effective_evidence_sids,
        "human_confirmed_evidence_sids": human_confirmed_evidence_sids,
        "fn_review_complete": fn_review_complete,
        "review_complete": review_complete,
        "human_evidence_status": human_evidence_status,
        "item_comment": st.session_state.get(item_comment_key(article_id, item_id), ""),
        "id_contract": record.get("id_contract", {}),
        "reviewed_at": utc_now(),
    }


# =============================================================================
# UI helpers
# =============================================================================
def baseline_status_display(status: str) -> Tuple[str, str]:
    mapping = {
        "reported": ("✅", "Reported"),
        "potentially_missing": ("❌", "Potentially missing"),
        "not_evaluated": ("⚪", "Not evaluated"),
    }
    return mapping.get(status, ("•", status or "Unknown"))


def saved_review_badge(review: Optional[Dict[str, Any]]) -> str:
    if not review:
        return ""
    if review.get("review_complete"):
        return " · 🧑✓"
    return " · 🧑…"


def corrected_status_text(review: Optional[Dict[str, Any]]) -> Optional[str]:
    if not review:
        return None
    status = review.get("human_evidence_status")
    return {
        "reported": "Human review: evidence present",
        "no_evidence_found": "Human review: no evidence found",
        "partially_reviewed_with_evidence": "Human review: partial, evidence confirmed",
        "partially_reviewed": "Human review: partial",
    }.get(status, "Human review saved")


def article_section_label(paragraph: Dict[str, Any]) -> str:
    sections = paragraph.get("predicted_sections", []) or []
    return " / ".join(str(x) for x in sections) if sections else "unclassified"


def format_article_option(record: Dict[str, Any]) -> str:
    return f"{record['article_id']} — {infer_title(record)}"


def feedback_download_bytes(store: Dict[str, Any]) -> bytes:
    return json.dumps(store, indent=2, ensure_ascii=False).encode("utf-8")


def _sentence_markup(
    sentence: Dict[str, Any],
    *,
    article_id: str,
    item_id: str,
    baseline_sid_set: set[str],
    added_sid_set: set[str],
) -> str:
    """Render one canonical sentence inside the continuous article view.

    IDs are used only to choose styling; they are deliberately not printed in
    the full-text view. The displayed text comes directly from the canonical
    sentence object in the precomputed export.
    """
    sid = str(sentence.get("sid", ""))
    safe_text = html.escape(str(sentence.get("text", "")))

    if sid in baseline_sid_set:
        decision = st.session_state.get(decision_key(article_id, item_id, sid), "unreviewed")
        if decision == "incorrect":
            return f'<span class="article-fp-evidence">{safe_text}</span>'
        return f'<mark class="article-model-evidence">{safe_text}</mark>'

    if sid in added_sid_set:
        return f'<mark class="article-human-evidence">{safe_text}</mark>'

    return safe_text


def build_full_article_html(
    record: Dict[str, Any],
    *,
    article_id: str,
    item_id: str,
    baseline_sid_set: set[str],
    added_sid_set: set[str],
) -> str:
    """Reconstruct the article as ordered, readable paragraphs.

    The main reading view intentionally contains no pnum/pid/sid labels.
    Sentence identifiers are exposed only in the separate human-evaluation
    selector below the article.
    """
    blocks: List[str] = []
    paragraphs = sorted(
        record.get("paragraphs", []) or [],
        key=lambda p: (p.get("pnum") is None, p.get("pnum", 0)),
    )

    for paragraph in paragraphs:
        sentences = paragraph.get("sentences", []) or []
        if sentences:
            rendered = " ".join(
                _sentence_markup(
                    s,
                    article_id=article_id,
                    item_id=item_id,
                    baseline_sid_set=baseline_sid_set,
                    added_sid_set=added_sid_set,
                )
                for s in sentences
            ).strip()
        else:
            rendered = html.escape(str(paragraph.get("text", "")).strip())

        if not rendered:
            continue

        raw_text = str(paragraph.get("text", "")).strip()
        # Short single-sentence paragraphs are usually article/section headings
        # in the exported corpus. Render them as headings but do not alter IDs.
        is_heading = (
            len(sentences) == 1
            and len(raw_text) <= 90
            and not re.search(r"[.!?]$", raw_text)
        ) or raw_text.lower() in {
            "title", "abstract", "introduction", "methods", "results",
            "discussion", "conclusion", "outcomes", "back matter",
            "sample size and analysis", "study population and procedures",
            "limitations of study", "comparison with other studies",
            "what is already known on this topic", "what this study adds",
        }

        if is_heading:
            blocks.append(f'<h4 class="article-heading">{rendered}</h4>')
        else:
            blocks.append(f'<p>{rendered}</p>')

    return '<div class="article-fulltext">' + "".join(blocks) + "</div>"


def paragraph_selector_label(paragraph: Dict[str, Any]) -> str:
    text = re.sub(r"\s+", " ", str(paragraph.get("text", "")).strip())
    excerpt = text[:105] + ("…" if len(text) > 105 else "")
    section = article_section_label(paragraph)
    return f"{section} · pnum={paragraph.get('pnum')} · pid={paragraph.get('pid')} · {excerpt}"


# =============================================================================
# Styling
# =============================================================================
st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1900px;
      }
      .small-note {font-size: .84rem; opacity: .78;}
      .sentence-legend {
        padding: .5rem .65rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: .55rem;
        margin-bottom: .6rem;
        font-size: .84rem;
      }
      .paragraph-meta {
        font-size: .76rem;
        opacity: .68;
        margin-top: .15rem;
      }
      .article-fulltext {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.02rem;
        line-height: 1.72;
      }
      .article-fulltext p {
        margin: 0 0 1.05rem 0;
      }
      .article-fulltext .article-heading {
        font-family: inherit;
        font-size: 1.08rem;
        margin: 1.3rem 0 .55rem 0;
      }
      .article-model-evidence {
        background: rgba(255, 214, 10, .35);
        padding: .05rem .12rem;
        border-radius: .18rem;
      }
      .article-human-evidence {
        background: rgba(74, 144, 226, .25);
        padding: .05rem .12rem;
        border-radius: .18rem;
      }
      .article-fp-evidence {
        text-decoration: line-through;
        text-decoration-thickness: 2px;
        opacity: .72;
      }
      div[data-testid="stButton"] > button {
        text-align: left;
        justify-content: flex-start;
        white-space: normal;
        height: auto;
        min-height: 2.35rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Initial state
# =============================================================================

# =============================================================================
# V5 page shell
# =============================================================================
if "feedback_store" not in st.session_state:
    st.session_state.feedback_store = load_feedback_store()
if "selected_item" not in st.session_state or st.session_state.selected_item not in ITEM_ORDER:
    st.session_state.selected_item = "12a" if "12a" in ITEM_ORDER else ITEM_ORDER[0]
if "live_record" not in st.session_state:
    st.session_state.live_record = None
if "live_record_hash" not in st.session_state:
    st.session_state.live_record_hash = None

UI_SECTIONS = ["Introduction", "Methods", "Results", "Discussion", "Other"]

# Import immutable precomputed studies into the persistent repository once.
if bootstrap_precomputed_records is not None:
    try:
        bootstrap_precomputed_records(APP_DIR, DEFAULT_DATA_PATH)
    except Exception:
        pass


def render_overview() -> None:
    st.title("CONSORT Evidence Extraction")
    st.write(
        "A prototype for turning randomized-trial reports into sentence-level CONSORT evidence, "
        "with a human review loop that can feed later prompt/prototype optimization."
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("### Input")
            st.markdown(
                "- Saved precomputed trial results, or\n"
                "- A new plain-text (`.txt`) trial report\n"
                "- OpenAI API credential for live processing"
            )
    with c2:
        with st.container(border=True):
            st.markdown("### Output")
            st.markdown(
                "- Full source article\n"
                "- CONSORT reporting/compliance view\n"
                "- Sentence-level extracted evidence\n"
                "- Human TP/FP/FN corrections\n- Cross-study evidence synthesis"
            )
    with c3:
        with st.container(border=True):
            st.markdown("### Human-in-the-loop")
            st.markdown(
                "Review every extracted sentence, reject false positives, and add missed source "
                "sentences without changing the article's canonical sentence IDs."
            )

    st.markdown("## How it works")
    steps = [
        ("1", "Preprocess", "TXT is split into ordered paragraphs and canonical sentences with stable pid/pnum/sid identifiers."),
        ("2", "Route", "A global article-level router assigns each paragraph to one or more trial-report sections."),
        ("3", "Extract", "Section-specific CONSORT extractors return item-to-sentence evidence pairs."),
        ("4", "Review", "The app shows the full paper, reporting status, and extracted evidence in one synchronized workspace."),
        ("5", "Correct", "Human sentence-level TP/FP/FN feedback is saved in a GEPA-friendly format for later optimization."),
        ("6", "Synthesize", "Accepted evidence is normalized into cached study profiles and organized into comparable evidence families."),
    ]
    for number, title, body in steps:
        a, b = st.columns([0.07, 0.93])
        a.markdown(f"### {number}")
        b.markdown(f"**{title}**  \n{body}")

    st.markdown("## Current prototype scope")
    st.markdown(
        "The live path supports `.txt` reports only. The current online pipeline runs preprocessing, "
        "section routing, article-context construction, and section-specific extraction. The verifier/recovery stage is not used."
    )


def status_counts_for_record(record: Dict[str, Any]) -> Dict[str, int]:
    counts = {"reported": 0, "potentially_missing": 0, "not_evaluated": 0}
    cidx = checklist_index(record)
    for item_id in ITEM_ORDER:
        status = str(cidx.get(item_id, {}).get("status", "not_evaluated"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def render_results_workspace(records: List[Dict[str, Any]], *, key_prefix: str) -> None:
    """Shared saved/live review workspace. Live results intentionally use this exact renderer."""
    if not records:
        st.info("No results available.")
        return

    record_map = article_index(records)
    if len(record_map) != len(records):
        st.error("Duplicate article IDs detected.")
        return

    article_ids = list(record_map)
    selected_article_id = st.selectbox(
        "Article",
        options=article_ids,
        format_func=lambda aid: format_article_option(record_map[aid]),
        key=f"{key_prefix}__article",
    )
    record = record_map[selected_article_id]

    warnings = list(record.get("id_warnings", []) or []) + validate_article_record(record)
    if warnings:
        with st.expander(f"Identifier warnings ({len(warnings)})"):
            for warning in warnings:
                st.warning(str(warning))

    baseline_by_item = baseline_evidence_by_item(record)
    checklist_by_item = checklist_index(record)
    s_index = sentence_index(record)

    counts = status_counts_for_record(record)
    article_reviews = [
        r for r in st.session_state.feedback_store.get("reviews", {}).values()
        if r.get("article_id") == selected_article_id and r.get("item_id") in ITEM_ORDER
    ]
    completed = sum(bool(r.get("review_complete")) for r in article_reviews)

    m1, m2, m3 = st.columns(3)
    m1.metric("Reported", counts.get("reported", 0))
    m2.metric("Potentially missing", counts.get("potentially_missing", 0))
    m3.metric("Human-reviewed", f"{completed}/{len(ITEM_ORDER)}")

    article_col, compliance_col, extraction_col = st.columns([1.55, 1.0, 1.25], gap="large")

    # Compliance first because item selection drives the other columns.
    with compliance_col:
        st.subheader("Compliance check")

        statuses_present = [
            s for s in ["reported", "potentially_missing", "not_evaluated"]
            if any(str(checklist_by_item.get(i, {}).get("status", "not_evaluated")) == s for i in ITEM_ORDER)
        ]
        default_statuses = [s for s in ["reported", "potentially_missing"] if s in statuses_present]
        if not default_statuses:
            default_statuses = statuses_present

        selected_statuses = st.multiselect(
            "Status",
            options=statuses_present,
            default=default_statuses,
            format_func=lambda x: baseline_status_display(x)[1],
            key=f"{key_prefix}__status_filter__{selected_article_id}",
        )
        selected_sections = st.multiselect(
            "Section",
            options=UI_SECTIONS,
            default=UI_SECTIONS,
            key=f"{key_prefix}__section_filter__{selected_article_id}",
        )
        query = st.text_input(
            "Search item",
            placeholder="e.g., randomization or 12a",
            key=f"{key_prefix}__item_search__{selected_article_id}",
        ).strip().lower()

        visible_items: List[str] = []
        for item_id in ITEM_ORDER:
            meta = ITEM_BY_ID[item_id]
            status = str(checklist_by_item.get(item_id, {}).get("status", "not_evaluated"))
            if selected_statuses and status not in selected_statuses:
                continue
            if selected_sections and meta["section"] not in selected_sections:
                continue
            haystack = f"{item_id} {meta['label']} {meta['section']}".lower()
            if query and query not in haystack:
                continue
            visible_items.append(item_id)

        if not visible_items:
            st.info("No items match the current filters.")
            selected_item_id = st.session_state.selected_item
        else:
            current = st.session_state.selected_item
            if current not in visible_items:
                current = visible_items[0]

            def _item_label(item_id: str) -> str:
                status = str(checklist_by_item.get(item_id, {}).get("status", "not_evaluated"))
                icon, _ = baseline_status_display(status)
                return f"{icon} {item_id} — {ITEM_BY_ID[item_id]['label']}{saved_review_badge(get_saved_review(selected_article_id, item_id))}"

            selected_item_id = st.radio(
                "CONSORT item",
                visible_items,
                index=visible_items.index(current),
                format_func=_item_label,
                label_visibility="collapsed",
                key=f"{key_prefix}__item_radio__{selected_article_id}",
            )
            st.session_state.selected_item = selected_item_id

        meta = ITEM_BY_ID[selected_item_id]
        status = str(checklist_by_item.get(selected_item_id, {}).get("status", "not_evaluated"))
        icon, label = baseline_status_display(status)
        st.markdown(f"**{selected_item_id} · {meta['section']}**")
        st.caption(meta["label"])
        st.write(f"{icon} **{label}** · {len(baseline_by_item.get(selected_item_id, []))} evidence sentence(s)")

        saved = get_saved_review(selected_article_id, selected_item_id)
        if saved:
            human_text = corrected_status_text(saved)
            if saved.get("review_complete"):
                st.success(human_text)
            else:
                st.warning(human_text)

    # Selected item fixed for the other columns.
    selected_item_id = st.session_state.selected_item
    baseline_rows = baseline_by_item.get(selected_item_id, [])
    initialise_review_widget_state(selected_article_id, selected_item_id, baseline_rows)
    baseline_sid_set = {str(x["sid"]) for x in baseline_rows}
    added_key = added_sids_key(selected_article_id, selected_item_id)
    added_sid_set = set(st.session_state.get(added_key, []))

    with article_col:

        st.markdown("#### Full article")

        full_article_html = build_full_article_html(
            record,
            article_id=selected_article_id,
            item_id=selected_item_id,
            baseline_sid_set=baseline_sid_set,
            added_sid_set=set(st.session_state.get(added_key, [])),
        )

        st.markdown(
            '<div class="sentence-legend">'
            '🟨 model evidence for selected item &nbsp;&nbsp; '
            '🟦 human-added missed evidence &nbsp;&nbsp; '
            'strikethrough = model evidence marked incorrect'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.container(height=720, border=True):
            st.markdown(full_article_html, unsafe_allow_html=True)

        st.markdown("#### Human evidence selection")
        st.caption(
            "Only this review area exposes paragraph/sentence identifiers. Choose a source paragraph, "
        )

        add_mode = st.toggle(
            f"Add missed evidence for {selected_item_id}",
            key=add_mode_key(selected_article_id, selected_item_id),
            help=(
                "Enable sentence selection for false-negative correction. The full article above remains unchanged "
                "except for evidence highlighting."
            ),
        )
        if add_mode:
            paragraphs = sorted(record.get("paragraphs", []) or [], key=lambda p: p.get("pnum", 0))
            paragraphs = [p for p in paragraphs if p.get("sentences")]
            if paragraphs:
                baseline_pnums = [r.get("pnum") for r in baseline_rows if r.get("pnum") is not None]
                default_pnum = baseline_pnums[0] if baseline_pnums else paragraphs[0].get("pnum")
                pnums = [p.get("pnum") for p in paragraphs]
                default_index = pnums.index(default_pnum) if default_pnum in pnums else 0
                paragraph = st.selectbox(
                    "Source paragraph",
                    paragraphs,
                    index=default_index,
                    format_func=paragraph_selector_label,
                    key=f"{key_prefix}__fn_paragraph__{selected_article_id}__{selected_item_id}",
                )
                for sentence in paragraph.get("sentences", []) or []:
                    sid = str(sentence.get("sid", ""))
                    text_value = str(sentence.get("text", ""))
                    if sid in baseline_sid_set:
                        marker = "🟨"
                    elif sid in added_sid_set:
                        marker = "🟦"
                    else:
                        marker = "+"
                    if st.button(
                        f"{marker} {sid} · {text_value}",
                        key=f"{key_prefix}__sentence_click__{selected_article_id}__{selected_item_id}__{sid}",
                        use_container_width=True,
                    ):
                        if sid in baseline_sid_set:
                            st.toast(f"{sid} is already a model prediction; review it in Extraction.")
                        else:
                            current_added = list(st.session_state.get(added_key, []))
                            if sid in current_added:
                                current_added = [x for x in current_added if x != sid]
                            else:
                                current_added.append(sid)
                            st.session_state[added_key] = current_added
                            st.rerun()

    with extraction_col:
        st.subheader("Extraction & review")
        meta = ITEM_BY_ID[selected_item_id]
        status = str(checklist_by_item.get(selected_item_id, {}).get("status", "not_evaluated"))
        icon, label = baseline_status_display(status)
        st.markdown(f"### {selected_item_id} — {meta['label']}")
        st.write(f"{icon} **{label}**")

        if not baseline_rows:
            st.info("No model evidence extracted for this item.")
        else:
            for idx, evidence in enumerate(baseline_rows, start=1):
                sid = str(evidence["sid"])
                with st.container(border=True):
                    st.markdown(f"**{idx}. {sid}**")
                    st.write(evidence.get("text", ""))
                    st.radio(
                        f"Review {sid}",
                        ["unreviewed", "correct", "incorrect"],
                        key=decision_key(selected_article_id, selected_item_id, sid),
                        horizontal=True,
                        format_func=lambda x: {"unreviewed": "Not reviewed", "correct": "✅ Correct", "incorrect": "❌ Incorrect"}[x],
                        label_visibility="collapsed",
                    )
                    st.text_input(
                        "Comment (optional)",
                        key=comment_key(selected_article_id, selected_item_id, sid),
                        placeholder="Reason for the correction",
                    )

        current_added_sids = [
            sid for sid in st.session_state.get(added_key, [])
            if sid in s_index and sid not in baseline_sid_set
        ]
        if current_added_sids:
            st.markdown("#### Added missed evidence")
            for sid in current_added_sids:
                sentence = s_index[sid]
                with st.container(border=True):
                    st.markdown(f"**🟦 {sid}**")
                    st.write(sentence.get("text", ""))
                    st.text_input(
                        "Comment (optional)",
                        key=fn_comment_key(selected_article_id, selected_item_id, sid),
                        placeholder="Why this sentence belongs to the item",
                    )
                    if st.button(
                        f"Remove {sid}",
                        key=f"{key_prefix}__remove_fn__{selected_article_id}__{selected_item_id}__{sid}",
                    ):
                        st.session_state[added_key] = [x for x in st.session_state.get(added_key, []) if x != sid]
                        st.rerun()

        st.checkbox(
            "Finished checking for missed evidence",
            key=fn_complete_key(selected_article_id, selected_item_id),
        )
        st.text_area(
            "Item comment (optional)",
            key=item_comment_key(selected_article_id, selected_item_id),
            height=70,
        )

        preview = build_review_payload(record, selected_item_id, baseline_rows)
        a, b, c = st.columns(3)
        a.metric("Reviewed", f"{len(baseline_rows)-len(preview['unreviewed_baseline_sids'])}/{len(baseline_rows)}")
        b.metric("FP", len(preview["false_positive_sids"]))
        c.metric("FN", len(preview["false_negative_added_sids"]))

        if st.button("Save review", type="primary", use_container_width=True, key=f"{key_prefix}__save_review"):
            rid = review_id(selected_article_id, selected_item_id)
            st.session_state.feedback_store.setdefault("reviews", {})[rid] = preview
            persist_feedback_store(st.session_state.feedback_store)
            st.success("Review saved.")

    # Cross-study synthesis is part of the selected study result rather than a
    # separate page. Profiles/families are cached on disk and reused.
    render_study_synthesis_panel(record, key_prefix=key_prefix)


def render_saved_results() -> None:
    st.title("Saved results")
    if load_repository_records is None:
        st.error(f"Study repository could not be imported: {SYNTHESIS_IMPORT_ERROR}")
        return

    try:
        bootstrap_precomputed_records(APP_DIR, DEFAULT_DATA_PATH)
        records = load_repository_records(APP_DIR)
    except Exception as exc:
        st.error(f"Could not load the study repository: {exc}")
        records = []

    if records:
        render_results_workspace(records, key_prefix="saved_v6")
    else:
        st.info("No saved studies are available yet. Upload a new .txt report or add the precomputed result JSON next to the app.")


def render_upload_new() -> None:
    st.title("Upload new")
    uploaded = st.file_uploader("Trial report (.txt)", type=["txt"], key="live_txt_upload_v6")
    if uploaded is None:
        st.caption("Upload a plain-text randomized-trial report to run the live extraction pipeline.")
        return
    if preprocess_txt is None or run_txt_pipeline is None:
        st.error(f"Live pipeline wrapper could not be imported: {LIVE_PIPELINE_IMPORT_ERROR}")
        return

    raw_text = uploaded.getvalue().decode("utf-8", errors="replace")
    article_id = Path(uploaded.name).stem
    try:
        preview = preprocess_txt(raw_text, article_id)
    except Exception as exc:
        st.error(f"TXT preprocessing failed: {exc}")
        return

    st.caption(f"{len(preview['paragraphs'])} paragraphs · {len(preview['sentences'])} sentences")

    cred_col, model_col = st.columns([1, 1])
    with cred_col:
        api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...", key="live_api_key_v6")
    with model_col:
        model_name = st.text_input("Model / deployment", value="openai/gpt-5.6-terra", key="live_model_v6")

    with st.expander("Pipeline settings", expanded=False):
        prototype_path_text = st.text_input("CONSORT prototype JSON", value=str(DEFAULT_PROTOTYPE_PATH), key="live_proto_v6")
        guideline_path_text = st.text_input("Annotation guideline JSON", value=str(DEFAULT_GUIDELINE_PATH), key="live_guideline_v6")
        router_path_text = st.text_input(
            "Optimized router state (optional)",
            value=str(DEFAULT_ROUTER_PATH) if DEFAULT_ROUTER_PATH.exists() else "",
            key="live_router_v6",
        )
        enable_article_context = st.checkbox("Build article context", value=True, key="live_context_v6")

    cfg = PipelineConfig(
        model=model_name,
        prototype_path=prototype_path_text.strip() or None,
        annotation_guideline_path=guideline_path_text.strip() or None,
        optimized_router_path=router_path_text.strip() or None,
        enable_article_context=enable_article_context,
    )
    assets = check_asset_paths(cfg)
    if not assets["prototype"]["exists"]:
        st.error("CONSORT prototype file is missing.")
        return
    if cfg.optimized_router_path and not assets["optimized_router"]["exists"]:
        st.error("Configured optimized router file is missing.")
        return

    if DEVELOPER_OPENAI_API_KEY.strip():
        st.caption("Development credential fallback is configured in the app file.")

    if st.button("Run extraction", type="primary", key="run_live_v6"):
        effective_api_key = api_key.strip() or DEVELOPER_OPENAI_API_KEY.strip()
        if not effective_api_key:
            st.error("Enter an OpenAI API key.")
        else:
            try:
                with st.status("Processing article…", expanded=True) as status:
                    def _progress(stage: str, detail: str) -> None:
                        status.write(f"{stage}: {detail}")
                    live_record = run_txt_pipeline(
                        text=raw_text,
                        article_id=article_id,
                        api_key=effective_api_key,
                        config=cfg,
                        progress_callback=_progress,
                    )
                    status.update(label="Extraction complete", state="complete", expanded=False)
                st.session_state.live_record = live_record
                st.session_state.live_record_hash = live_record.get("content_hash")
                LIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
                live_path = LIVE_RUN_DIR / f"{live_record['run_id']}.json"
                with open(live_path, "w", encoding="utf-8") as f:
                    json.dump(live_record, f, ensure_ascii=False, indent=2)
                if upsert_study_record is not None:
                    upsert_study_record(APP_DIR, live_record, source="live_upload")
                st.success("Extraction saved. This study is now available under Saved results.")
            except Exception as exc:
                st.exception(exc)

    current = st.session_state.live_record
    if current is not None and st.session_state.live_record_hash == preview.get("content_hash"):
        st.divider()
        # IMPORTANT: live output uses the exact same renderer as Saved Results.
        render_results_workspace([current], key_prefix="live")
        st.download_button(
            "Download result JSON",
            data=json.dumps(current, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{current['run_id']}.json",
            mime="application/json",
            key="download_live_json_v6",
        )


def _synthesis_api_key(user_value: str) -> str:
    return str(user_value or "").strip() or DEVELOPER_OPENAI_API_KEY.strip()


def _families_for_study(family_store: Dict[str, Any], article_id: str) -> List[Tuple[str, Dict[str, Any], str]]:
    """Return (family_id, family, question_id) memberships for one study."""
    memberships: List[Tuple[str, Dict[str, Any], str]] = []
    for family_id, family in (family_store.get("families", {}) or {}).items():
        for member in family.get("members", []) or []:
            if str(member.get("article_id", "")) == str(article_id):
                memberships.append((str(family_id), family, str(member.get("question_id", ""))))
    return memberships


def _profile_display_row(profile: Dict[str, Any], question: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    question = question or {}

    def field_value(key: str) -> str:
        obj = profile.get(key, {}) or {}
        if isinstance(obj, dict):
            return str(obj.get("value") or "Not available")
        return str(obj or "Not available")

    comp = profile.get("consort_completeness", {}) or {}
    return {
        "Study": profile.get("study_label") or profile.get("article_id"),
        "Sample Size": (profile.get("sample_size", {}) or {}).get("display", "Not available"),
        "Population": question.get("population") or field_value("population"),
        "Intervention": question.get("intervention") or field_value("intervention"),
        "Comparator": question.get("comparator") or field_value("comparator"),
        "Duration / Follow-up": field_value("duration_follow_up"),
        "Primary Outcome": question.get("primary_outcome") or field_value("primary_outcome"),
        "Effect / Result": field_value("effect_result"),
        "Safety": field_value("safety"),
        "CONSORT Completeness": comp.get("display", "Not available"),
    }


def render_study_synthesis_panel(record: Dict[str, Any], *, key_prefix: str) -> None:
    """Render cached study profile + its cross-study evidence family in-place."""
    st.divider()
    st.subheader("Evidence synthesis")

    if ensure_study_profile is None:
        st.error(f"Synthesis module could not be imported: {SYNTHESIS_IMPORT_ERROR}")
        return

    article_id = str(record.get("article_id", ""))
    try:
        state, cached_profile, _ = profile_cache_state(
            APP_DIR, record, st.session_state.feedback_store
        )
    except Exception as exc:
        st.error(f"Could not inspect the evidence profile: {exc}")
        return

    # Generate only when absent/stale. Once current, this block disappears and
    # the saved profile is reused without another LLM call.
    if state != "current":
        status_label = "not generated" if state == "missing" else "out of date after evidence changes"
        st.caption(f"Study evidence profile: {status_label}.")
        with st.expander("Generate evidence profile", expanded=True):
            k1, k2 = st.columns([1, 1])
            with k1:
                synth_key = st.text_input(
                    "OpenAI API key",
                    type="password",
                    placeholder="Uses development fallback if configured",
                    key=f"{key_prefix}__profile_key__{article_id}",
                )
            with k2:
                synth_model = st.text_input(
                    "Model / deployment",
                    value=DEFAULT_SYNTHESIS_MODEL,
                    key=f"{key_prefix}__profile_model__{article_id}",
                )
            if st.button(
                "Generate / refresh profile",
                type="primary",
                key=f"{key_prefix}__profile_generate__{article_id}",
            ):
                effective_key = _synthesis_api_key(synth_key)
                if not effective_key:
                    st.error("Enter an OpenAI API key or configure the development fallback in the app file.")
                else:
                    try:
                        with st.status("Building study profile and checking related evidence…", expanded=True) as status:
                            profile, generated = ensure_study_profile(
                                APP_DIR,
                                record,
                                st.session_state.feedback_store,
                                api_key=effective_key,
                                model=synth_model,
                                force=False,
                            )
                            status.write("Study evidence profile saved." if generated else "Cached profile reused.")
                            stats = assign_profile_questions(
                                APP_DIR,
                                profile,
                                api_key=effective_key,
                                model=synth_model,
                                force_reassign=generated,
                            )
                            status.write(
                                f"Family check: {stats.get('same_family', 0)} matched, "
                                f"{stats.get('new_family', 0)} new, {stats.get('pending', 0)} needs review."
                            )
                            status.update(label="Evidence synthesis ready", state="complete", expanded=False)
                        st.rerun()
                    except Exception as exc:
                        st.exception(exc)
        # If a stale cached profile exists, do not present it as current evidence.
        return

    profile = cached_profile
    if not profile:
        st.info("No study evidence profile is available.")
        return

    # ------------------------------------------------------------------
    # Current study profile
    # ------------------------------------------------------------------
    questions = profile.get("evidence_questions", []) or []
    selected_question: Optional[Dict[str, Any]] = None
    if questions:
        if len(questions) == 1:
            selected_question = questions[0]
        else:
            qids = [str(q.get("question_id", "")) for q in questions]
            selected_qid = st.selectbox(
                "Evidence question",
                options=qids,
                format_func=lambda qid: next(
                    (
                        f"{qid} — {q.get('intervention', '')} vs {q.get('comparator', '')}: {q.get('primary_outcome', '')}"
                        for q in questions if str(q.get('question_id', '')) == qid
                    ),
                    qid,
                ),
                key=f"{key_prefix}__profile_question__{article_id}",
            )
            selected_question = next(
                (q for q in questions if str(q.get("question_id", "")) == selected_qid),
                questions[0],
            )

    profile_tab, family_tab = st.tabs(["Study profile", "Related evidence"])
    with profile_tab:
        st.dataframe([_profile_display_row(profile, selected_question)], use_container_width=True, hide_index=True)
        objective = (profile.get("objective", {}) or {}).get("value")
        if objective:
            st.markdown(f"**Study objective:** {objective}")

    # ------------------------------------------------------------------
    # Family membership / comparison for the selected study
    # ------------------------------------------------------------------
    with family_tab:
        family_store = load_family_store(APP_DIR)
        profiles = load_all_profiles(APP_DIR)
        memberships = _families_for_study(family_store, article_id)
        pending = [
            p for p in (family_store.get("pending", []) or [])
            if str(p.get("article_id", "")) == article_id
        ]

        if pending:
            st.warning("One or more evidence questions need review before family assignment.")
            family_ids = sorted((family_store.get("families", {}) or {}).keys())
            for p in pending:
                qid = str(p.get("question_id", ""))
                q = p.get("question", {}) or {}
                with st.container(border=True):
                    st.markdown(f"**{qid}: {q.get('intervention', 'Not available')} vs {q.get('comparator', 'Not available')}**")
                    st.caption(str(p.get("reason", "Family compatibility needs human review.")))
                    action = st.radio(
                        "Decision",
                        ["existing_family", "new_family"],
                        format_func=lambda x: "Add to existing family" if x == "existing_family" else "Create new family",
                        horizontal=True,
                        key=f"{key_prefix}__pending_action__{article_id}__{qid}",
                    )
                    chosen_family = None
                    if action == "existing_family" and family_ids:
                        chosen_family = st.selectbox(
                            "Family",
                            family_ids,
                            format_func=lambda fid: f"{fid} — {family_store['families'][fid].get('family_label', fid)}",
                            key=f"{key_prefix}__pending_family__{article_id}__{qid}",
                        )
                    if st.button("Apply family decision", key=f"{key_prefix}__pending_apply__{article_id}__{qid}"):
                        try:
                            resolve_pending_assignment(
                                APP_DIR,
                                article_id=article_id,
                                question_id=qid,
                                action=action if (action != "existing_family" or chosen_family) else "new_family",
                                family_id=chosen_family,
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

        if not memberships:
            if not pending:
                st.info("This study has no assigned evidence family yet.")
            return

        # If the profile has multiple evidence questions/families, default to the
        # membership matching the selected question above.
        if selected_question:
            selected_qid = str(selected_question.get("question_id", ""))
            preferred = [m for m in memberships if m[2] == selected_qid]
        else:
            preferred = []
        choices = preferred or memberships

        if len(choices) == 1:
            selected_family_id, family, selected_family_qid = choices[0]
        else:
            membership_keys = [f"{fid}::{qid}" for fid, _, qid in choices]
            selected_membership = st.selectbox(
                "Evidence family",
                membership_keys,
                format_func=lambda k: next(
                    f"{fid} — {fam.get('family_label', fid)}"
                    for fid, fam, qid in choices if f"{fid}::{qid}" == k
                ),
                key=f"{key_prefix}__family_membership__{article_id}",
            )
            selected_family_id, family, selected_family_qid = next(
                (fid, fam, qid) for fid, fam, qid in choices
                if f"{fid}::{qid}" == selected_membership
            )

        st.markdown(f"**{family.get('topic_label', 'Evidence topic')}**")
        st.caption(f"{selected_family_id} · {family.get('family_label', selected_family_id)}")

        rows = topic_table_rows(family_store, family.get("topic_id", ""), profiles)
        if rows:
            display_rows = [
                {("Article ID" if k == "Study ID" else k): v for k, v in row.items()}
                for row in rows
            ]
            st.dataframe(display_rows, use_container_width=True, hide_index=True)
            if len(rows) > 1:
                st.caption(
                    "Studies in this topic may still differ on dose, route, or comparator — "
                    "see the Family column for which ones share the same strict evidence question "
                    "(the same group used for cross-study synthesis below)."
                )

        summary = family.get("summary")
        if summary:
            st.markdown("#### Cross-study synthesis")
            st.write(summary.get("summary", ""))
            heterogeneity = summary.get("heterogeneity", []) or []
            gaps = summary.get("evidence_gaps", []) or []
            if heterogeneity:
                st.markdown("**Heterogeneity:** " + "; ".join(str(x) for x in heterogeneity))
            if gaps:
                st.markdown("**Evidence gaps:** " + "; ".join(str(x) for x in gaps))

        with st.expander("Generate / refresh narrative", expanded=False):
            nk1, nk2 = st.columns([1, 1])
            with nk1:
                narrative_key = st.text_input(
                    "OpenAI API key",
                    type="password",
                    placeholder="Uses development fallback if configured",
                    key=f"{key_prefix}__family_summary_key__{article_id}__{selected_family_id}",
                )
            with nk2:
                narrative_model = st.text_input(
                    "Model / deployment",
                    value=DEFAULT_SYNTHESIS_MODEL,
                    key=f"{key_prefix}__family_summary_model__{article_id}__{selected_family_id}",
                )
            if st.button(
                "Generate / refresh synthesis",
                key=f"{key_prefix}__family_summary_generate__{article_id}__{selected_family_id}",
            ):
                effective_key = _synthesis_api_key(narrative_key)
                if not effective_key:
                    st.error("Enter an OpenAI API key or configure the development fallback.")
                else:
                    try:
                        generate_family_summary(
                            APP_DIR,
                            selected_family_id,
                            api_key=effective_key,
                            model=narrative_model,
                        )
                        st.rerun()
                    except Exception as exc:
                        st.exception(exc)


# Sidebar navigation
with st.sidebar:
    st.markdown("# PRISM")
    st.markdown("CONSORT Evidence extraction and synthesis tool")
    page = st.radio(
        "Navigation",
        ["Task setting", "Saved results", "Upload new"],
        label_visibility="collapsed",
        key="nav_v6",
    )
    st.divider()
    n_reviews = len(st.session_state.feedback_store.get("reviews", {}))
    st.caption(f"Human reviews: {n_reviews}")
    st.download_button(
        "Download feedback",
        data=feedback_download_bytes(st.session_state.feedback_store),
        file_name="human_feedback_store.json",
        mime="application/json",
        use_container_width=True,
        key="sidebar_feedback_download_v6",
    )
    if repository_paths is not None:
        family_path = repository_paths(APP_DIR)["families"]
        if family_path.exists():
            st.download_button(
                "Download evidence families",
                data=family_path.read_bytes(),
                file_name="evidence_families.json",
                mime="application/json",
                use_container_width=True,
                key="sidebar_family_download_v6",
            )

if page == "Task setting":
    render_overview()
elif page == "Saved results":
    render_saved_results()
elif page == "Upload new":
    render_upload_new()
