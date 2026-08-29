"""Cross-study evidence synthesis support for CONSORT Demo v6.

This module sits AFTER sentence-level CONSORT extraction/human review.
It does not read the full article to invent synthesis fields. Study profiles are
constructed only from the effective CONSORT evidence set:

    baseline extracted evidence
        + human-added FN evidence
        - human-rejected FP evidence

Profiles are cached on disk using an input hash, so unchanged studies do not
call the LLM again. Evidence-family membership is question-based rather than
objective-string clustering: objective + population + intervention + comparator
+ primary outcome are compared explicitly.

No quantitative meta-analysis is performed in v6.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import dspy
except Exception:  # pragma: no cover
    dspy = None


SYNTHESIS_SCHEMA_VERSION = "consort_synthesis_v1"
PROFILE_VERSION = "study_profile_v1"
FAMILY_VERSION = "evidence_family_v1"

# UI/extraction scope intentionally excludes 1a/1b.
CONSORT_ITEM_ORDER = [
    "2a", "2b", "3a", "3b", "4a", "4b", "5", "6a", "6b", "7a", "7b",
    "8a", "8b", "9", "10", "11a", "11b", "12a", "12b", "13a", "13b",
    "14a", "14b", "15", "16", "17a", "17b", "18", "19", "20", "21",
    "22", "23", "24", "25",
]

PROFILE_RELEVANT_ITEMS = [
    "2b", "3a", "4a", "4b", "5", "6a", "7a", "13a", "13b", "14a",
    "15", "16", "17a", "17b", "18", "19",
]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "study")).strip("._")
    return value or "study"


def _json_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return copy.deepcopy(default)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return copy.deepcopy(default)


def extract_json_object(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    text = str(text or "").strip()
    if not text:
        raise ValueError("Empty LLM output")
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Could not find a valid JSON object in LLM output")


# -----------------------------------------------------------------------------
# Persistent study repository
# -----------------------------------------------------------------------------
def repository_paths(base_dir: Path) -> Dict[str, Path]:
    root = Path(base_dir) / "study_store"
    return {
        "root": root,
        "records": root / "records",
        "profiles": root / "profiles",
        "index": root / "index.json",
        "families": root / "evidence_families.json",
    }


def _empty_index() -> Dict[str, Any]:
    return {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "studies": {},
    }


def upsert_study_record(base_dir: Path, record: Dict[str, Any], source: str) -> Path:
    """Store one article record using article_id as repository identity.

    Re-uploading the same article ID updates its repository record rather than
    creating duplicate studies. The benchmark source JSON itself is never edited.
    """
    paths = repository_paths(base_dir)
    paths["records"].mkdir(parents=True, exist_ok=True)
    article_id = str(record.get("article_id", "")).strip()
    if not article_id:
        raise ValueError("Study record is missing article_id")

    payload = copy.deepcopy(record)
    payload["repository_metadata"] = {
        "source": str(source),
        "updated_at": utc_now(),
        "content_hash": record.get("content_hash"),
        "pipeline_stage": record.get("pipeline_stage"),
    }

    record_path = paths["records"] / f"{_safe_name(article_id)}.json"
    _atomic_write_json(record_path, payload)

    index = _read_json(paths["index"], _empty_index())
    index.setdefault("studies", {})[article_id] = {
        "article_id": article_id,
        "record_file": record_path.name,
        "source": str(source),
        "pipeline_stage": record.get("pipeline_stage"),
        "content_hash": record.get("content_hash"),
        "updated_at": utc_now(),
    }
    index["updated_at"] = utc_now()
    _atomic_write_json(paths["index"], index)
    return record_path


def bootstrap_precomputed_records(base_dir: Path, precomputed_json_path: Path) -> int:
    """Import immutable precomputed records into the repository once.

    Existing repository records are not overwritten, so a later live extraction
    of the same article can remain the active saved result.
    """
    if not Path(precomputed_json_path).exists():
        return 0
    with open(precomputed_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Precomputed result file must contain a JSON list")

    paths = repository_paths(base_dir)
    existing_index = _read_json(paths["index"], _empty_index())
    existing = set(existing_index.get("studies", {}))
    n = 0
    for record in records:
        article_id = str(record.get("article_id", "")).strip()
        if not article_id or article_id in existing:
            continue
        upsert_study_record(base_dir, record, source="precomputed_test")
        n += 1
    return n


def load_repository_records(base_dir: Path) -> List[Dict[str, Any]]:
    paths = repository_paths(base_dir)
    index = _read_json(paths["index"], _empty_index())
    records: List[Dict[str, Any]] = []
    for article_id, meta in sorted(index.get("studies", {}).items()):
        path = paths["records"] / str(meta.get("record_file", f"{_safe_name(article_id)}.json"))
        if not path.exists():
            continue
        obj = _read_json(path, None)
        if isinstance(obj, dict):
            records.append(obj)
    return records


# -----------------------------------------------------------------------------
# Human-corrected effective evidence and deterministic completeness
# -----------------------------------------------------------------------------
def _sentence_index(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(s.get("sid")): s for s in record.get("sentences", []) or [] if s.get("sid")}


def _baseline_by_item(record: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {i: [] for i in CONSORT_ITEM_ORDER}
    for e in record.get("final_extractions", []) or []:
        item = str(e.get("item", ""))
        if item in out:
            out[item].append(e)
    return out


def _checklist_index(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(x.get("item")): x for x in record.get("checklist", []) or []}


def _review_for(feedback_store: Dict[str, Any], article_id: str, item: str) -> Optional[Dict[str, Any]]:
    return (feedback_store or {}).get("reviews", {}).get(f"{article_id}::{item}")


def effective_evidence_by_item(
    record: Dict[str, Any],
    feedback_store: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return model evidence after explicit human corrections when available."""
    feedback_store = feedback_store or {"reviews": {}}
    article_id = str(record.get("article_id", ""))
    sidx = _sentence_index(record)
    baseline = _baseline_by_item(record)
    out: Dict[str, List[Dict[str, Any]]] = {}

    for item in CONSORT_ITEM_ORDER:
        review = _review_for(feedback_store, article_id, item)
        if review:
            # The app's effective set retains unreviewed baseline evidence, removes
            # explicit FPs, and adds human FNs. This is the correct provisional set
            # for downstream synthesis; fully reviewed items become authoritative.
            sids = [str(x) for x in review.get("effective_evidence_sids", []) or []]
        else:
            sids = [str(x.get("sid")) for x in baseline.get(item, []) if x.get("sid")]

        rows = []
        for sid in dict.fromkeys(sids):
            if sid not in sidx:
                continue
            s = sidx[sid]
            rows.append(
                {
                    "item": item,
                    "sid": sid,
                    "pnum": s.get("pnum"),
                    "pid": str(s.get("pid", "")),
                    "text": str(s.get("text", "")),
                    "human_modified": bool(review),
                    "review_complete": bool(review.get("review_complete")) if review else False,
                }
            )
        out[item] = rows
    return out


def consort_completeness(
    record: Dict[str, Any],
    feedback_store: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic reporting completeness over evaluated UI-scope items."""
    feedback_store = feedback_store or {"reviews": {}}
    article_id = str(record.get("article_id", ""))
    cidx = _checklist_index(record)
    effective = effective_evidence_by_item(record, feedback_store)

    statuses: Dict[str, str] = {}
    for item in CONSORT_ITEM_ORDER:
        baseline_status = str(cidx.get(item, {}).get("status", "not_evaluated"))
        review = _review_for(feedback_store, article_id, item)
        if review:
            confirmed = list(review.get("human_confirmed_evidence_sids", []) or [])
            if confirmed:
                status = "reported"
            elif review.get("review_complete"):
                status = "reported" if effective.get(item) else "potentially_missing"
            else:
                # Partial review does not erase a baseline status unless evidence
                # has already been explicitly confirmed.
                status = baseline_status
        else:
            status = baseline_status
        statuses[item] = status

    evaluated = [i for i, status in statuses.items() if status != "not_evaluated"]
    reported = [i for i in evaluated if statuses[i] == "reported"]
    missing = [i for i in evaluated if statuses[i] == "potentially_missing"]
    pct = (len(reported) / len(evaluated)) if evaluated else None
    return {
        "reported": len(reported),
        "evaluated": len(evaluated),
        "potentially_missing": len(missing),
        "percentage": pct,
        "display": f"{len(reported)}/{len(evaluated)} ({pct:.1%})" if pct is not None else "Not available",
        "reported_items": reported,
        "potentially_missing_items": missing,
        "item_status": statuses,
    }


def build_profile_input(
    record: Dict[str, Any],
    feedback_store: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    effective = effective_evidence_by_item(record, feedback_store)
    evidence_rows = []
    for item in PROFILE_RELEVANT_ITEMS:
        for row in effective.get(item, []):
            evidence_rows.append(row)
    completeness = consort_completeness(record, feedback_store)
    payload = {
        "article_id": str(record.get("article_id", "")),
        "pipeline_stage": record.get("pipeline_stage"),
        "evidence": evidence_rows,
        "consort_completeness": completeness,
    }
    payload["input_hash"] = _json_hash(payload)
    return payload


# -----------------------------------------------------------------------------
# DSPy profile / family reasoning
# -----------------------------------------------------------------------------
if dspy is not None:
    class StudyEvidenceProfileSignature(dspy.Signature):
        """Create a structured study profile ONLY from supplied CONSORT evidence.

        Return valid JSON only in output_json with this shape:
        {
          "study_label": "short human-readable label",
          "objective": FIELD,
          "sample_size": {
             "randomized": "string or null",
             "analyzed": "string or null",
             "planned": "string or null",
             "display": "concise display string",
             "status": "supported|ambiguous|unavailable",
             "source_items": [...], "source_sids": [...]
          },
          "population": FIELD,
          "intervention": FIELD,
          "comparator": FIELD,
          "duration_follow_up": FIELD,
          "primary_outcome": FIELD,
          "effect_result": FIELD,
          "safety": FIELD,
          "evidence_questions": [
            {
              "question_id": "Q1",
              "objective": "...",
              "population": "...",
              "intervention": "...",
              "comparator": "...",
              "primary_outcome": "...",
              "source_items": [...], "source_sids": [...]
            }
          ]
        }

        FIELD = {"value":"...", "status":"supported|ambiguous|unavailable",
                 "source_items":[...], "source_sids":[...]}

        Rules:
        - Never use outside knowledge or infer facts absent from the evidence.
        - If evidence is insufficient, status=unavailable and value="Not available from extracted CONSORT evidence".
        - Objective should be grounded primarily in item 2b.
        - Actual sample size should use participant-flow/analysis evidence (13a/13b/16), NOT item 7a alone.
        - Item 7a may support planned sample size only.
        - Population primarily uses 4a/4b/15; intervention/comparator use item 5.
        - Duration/follow-up may use 5, 6a, and 14a.
        - Primary outcome uses 6a; effect/result uses 17a/17b (18 only for ancillary findings).
        - Safety uses 19, or outcome-result evidence only when safety is itself an explicitly defined outcome.
        - Preserve multi-arm comparisons. Create more than one evidence question only when the supplied evidence clearly supports distinct comparisons.
        - Every source SID/item must exist in the supplied evidence. Do not cite evidence merely because it seems plausible.
        - evidence_questions are normalized research questions for family matching. Do not create a question when objective, intervention, or primary outcome is unavailable.
        """
        article_id = dspy.InputField()
        evidence_json = dspy.InputField()
        output_json = dspy.OutputField(desc="Valid JSON object only")


    class EvidenceFamilyAssignmentSignature(dspy.Signature):
        """Compare one normalized evidence question against existing strict families.

        Return valid JSON only:
        {
          "decision": "same_family|related_but_distinct|no_related_family|needs_review",
          "family_id": "EF... or null",
          "related_family_id": "EF... or null",
          "topic_label": "short broader evidence topic",
          "family_label": "short strict family label for a new family, else existing label",
          "compatibility": {
             "objective": "same|compatible|different|uncertain",
             "population": "same|compatible|different|uncertain",
             "intervention": "same|compatible|different|uncertain",
             "comparator": "same|compatible|different|uncertain",
             "primary_outcome": "same|compatible|different|uncertain"
          },
          "reason": "brief explanation"
        }

        Strict family membership means the studies address the same estimand-like
        evidence question, not merely the same broad topic. Different intervention,
        comparator, clinically distinct population, or primary outcome normally
        means related_but_distinct rather than same_family. If information is
        insufficient, use needs_review. Use no_related_family only when no current
        topic/family is meaningfully related.
        """
        question_json = dspy.InputField()
        existing_families_json = dspy.InputField()
        output_json = dspy.OutputField(desc="Valid JSON object only")


    class FamilyNarrativeSignature(dspy.Signature):
        """Summarize a family using ONLY supplied structured study profiles.

        Return JSON only: {"summary":"2-4 sentence cautious cross-study synthesis",
        "heterogeneity":["..."], "evidence_gaps":["..."]}.
        Do not pool effects, compute new statistics, or claim causal/generalizable
        conclusions beyond the supplied profiles.
        """
        family_json = dspy.InputField()
        member_profiles_json = dspy.InputField()
        output_json = dspy.OutputField(desc="Valid JSON object only")
else:  # pragma: no cover
    StudyEvidenceProfileSignature = None
    EvidenceFamilyAssignmentSignature = None
    FamilyNarrativeSignature = None


def configure_synthesis_lm(
    api_key: str,
    model: str,
    *,
    reasoning_effort: str = "medium",
    max_tokens: int = 16000,
) -> "dspy.LM":
    """Build a DSPy LM for synthesis calls.

    Returns the LM instead of calling `dspy.configure(lm=...)` because DSPy's
    global configure is locked to whichever thread calls it first, and
    Streamlit reruns the script on a new thread per interaction. Callers
    should use `with dspy.context(lm=configure_synthesis_lm(...)):`, which
    DSPy allows from any thread.
    """
    if dspy is None:
        raise ImportError("DSPy is required for synthesis. Install with `pip install dspy`.")
    if not str(api_key or "").strip():
        raise ValueError("An OpenAI API key is required when a new synthesis LLM call is needed.")
    kwargs: Dict[str, Any] = {
        "api_key": str(api_key).strip(),
        "temperature": 1.0,
        "max_tokens": max_tokens,
        "max_retries": 6,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        lm = dspy.LM(model, **kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        lm = dspy.LM(model, **kwargs)
    return lm


def _normalize_field(obj: Any) -> Dict[str, Any]:
    default_text = "Not available from extracted CONSORT evidence"
    if not isinstance(obj, dict):
        return {"value": default_text, "status": "unavailable", "source_items": [], "source_sids": []}
    status = str(obj.get("status", "unavailable"))
    if status not in {"supported", "ambiguous", "unavailable"}:
        status = "unavailable"
    value = obj.get("value")
    if value is None or not str(value).strip():
        value = default_text if status == "unavailable" else "Ambiguous from extracted CONSORT evidence"
    return {
        "value": str(value),
        "status": status,
        "source_items": [str(x) for x in obj.get("source_items", []) or []],
        "source_sids": [str(x) for x in obj.get("source_sids", []) or []],
    }


def _clean_profile_sources(profile: Dict[str, Any], valid_items: set[str], valid_sids: set[str]) -> None:
    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if "source_items" in x:
                x["source_items"] = [str(v) for v in x.get("source_items", []) or [] if str(v) in valid_items]
            if "source_sids" in x:
                x["source_sids"] = [str(v) for v in x.get("source_sids", []) or [] if str(v) in valid_sids]
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(profile)


def normalize_profile(raw: Dict[str, Any], profile_input: Dict[str, Any]) -> Dict[str, Any]:
    fields = [
        "objective", "population", "intervention", "comparator", "duration_follow_up",
        "primary_outcome", "effect_result", "safety",
    ]
    out: Dict[str, Any] = {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "profile_version": PROFILE_VERSION,
        "article_id": profile_input["article_id"],
        "input_hash": profile_input["input_hash"],
        "generated_at": utc_now(),
        "study_label": str(raw.get("study_label") or profile_input["article_id"]),
    }
    for key in fields:
        out[key] = _normalize_field(raw.get(key))

    ss = raw.get("sample_size") if isinstance(raw.get("sample_size"), dict) else {}
    ss_status = str(ss.get("status", "unavailable"))
    if ss_status not in {"supported", "ambiguous", "unavailable"}:
        ss_status = "unavailable"
    out["sample_size"] = {
        "randomized": ss.get("randomized"),
        "analyzed": ss.get("analyzed"),
        "planned": ss.get("planned"),
        "display": str(ss.get("display") or "Not available from extracted CONSORT evidence"),
        "status": ss_status,
        "source_items": [str(x) for x in ss.get("source_items", []) or []],
        "source_sids": [str(x) for x in ss.get("source_sids", []) or []],
    }

    questions = []
    for i, q in enumerate(raw.get("evidence_questions", []) or [], start=1):
        if not isinstance(q, dict):
            continue
        objective = str(q.get("objective", "")).strip()
        intervention = str(q.get("intervention", "")).strip()
        outcome = str(q.get("primary_outcome", "")).strip()
        if not objective or not intervention or not outcome:
            continue
        questions.append(
            {
                "question_id": str(q.get("question_id") or f"Q{i}"),
                "objective": objective,
                "population": str(q.get("population", "Not available")).strip() or "Not available",
                "intervention": intervention,
                "comparator": str(q.get("comparator", "Not available")).strip() or "Not available",
                "primary_outcome": outcome,
                "source_items": [str(x) for x in q.get("source_items", []) or []],
                "source_sids": [str(x) for x in q.get("source_sids", []) or []],
            }
        )
    out["evidence_questions"] = questions
    out["consort_completeness"] = profile_input["consort_completeness"]

    valid_items = {str(x.get("item")) for x in profile_input.get("evidence", [])}
    valid_sids = {str(x.get("sid")) for x in profile_input.get("evidence", [])}
    _clean_profile_sources(out, valid_items, valid_sids)
    return out


def profile_path(base_dir: Path, article_id: str) -> Path:
    return repository_paths(base_dir)["profiles"] / f"{_safe_name(article_id)}.json"


def load_profile(base_dir: Path, article_id: str) -> Optional[Dict[str, Any]]:
    obj = _read_json(profile_path(base_dir, article_id), None)
    return obj if isinstance(obj, dict) else None


def load_all_profiles(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    paths = repository_paths(base_dir)
    result: Dict[str, Dict[str, Any]] = {}
    if not paths["profiles"].exists():
        return result
    for path in sorted(paths["profiles"].glob("*.json")):
        obj = _read_json(path, None)
        if isinstance(obj, dict) and obj.get("article_id"):
            result[str(obj["article_id"])] = obj
    return result


def profile_cache_state(
    base_dir: Path,
    record: Dict[str, Any],
    feedback_store: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    inp = build_profile_input(record, feedback_store)
    cached = load_profile(base_dir, str(record.get("article_id", "")))
    if cached and cached.get("input_hash") == inp["input_hash"]:
        return "current", cached, inp
    if cached:
        return "stale", cached, inp
    return "missing", None, inp


def ensure_study_profile(
    base_dir: Path,
    record: Dict[str, Any],
    feedback_store: Optional[Dict[str, Any]],
    *,
    api_key: str,
    model: str,
    force: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    state, cached, inp = profile_cache_state(base_dir, record, feedback_store)
    if state == "current" and cached is not None and not force:
        return cached, False

    lm = configure_synthesis_lm(api_key, model)
    predictor = dspy.Predict(StudyEvidenceProfileSignature)
    evidence_payload = {
        "article_id": inp["article_id"],
        "evidence": inp["evidence"],
        "consort_completeness": inp["consort_completeness"],
    }
    with dspy.context(lm=lm):
        pred = predictor(
            article_id=inp["article_id"],
            evidence_json=json.dumps(evidence_payload, ensure_ascii=False),
        )
    raw = extract_json_object(pred.output_json)
    profile = normalize_profile(raw, inp)
    _atomic_write_json(profile_path(base_dir, inp["article_id"]), profile)
    return profile, True


# -----------------------------------------------------------------------------
# Evidence families
# -----------------------------------------------------------------------------
def _empty_family_store() -> Dict[str, Any]:
    return {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "family_version": FAMILY_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "families": {},
        "pending": [],
    }


def load_family_store(base_dir: Path) -> Dict[str, Any]:
    return _read_json(repository_paths(base_dir)["families"], _empty_family_store())


def save_family_store(base_dir: Path, store: Dict[str, Any]) -> None:
    store["updated_at"] = utc_now()
    _atomic_write_json(repository_paths(base_dir)["families"], store)


def _next_numeric_id(existing: Iterable[str], prefix: str) -> str:
    nums = []
    for value in existing:
        m = re.fullmatch(re.escape(prefix) + r"(\d+)", str(value))
        if m:
            nums.append(int(m.group(1)))
    return f"{prefix}{max(nums, default=0)+1:03d}"


def _family_public_view(store: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for family in store.get("families", {}).values():
        rows.append(
            {
                "family_id": family.get("family_id"),
                "topic_id": family.get("topic_id"),
                "topic_label": family.get("topic_label"),
                "family_label": family.get("family_label"),
                "signature": family.get("signature", {}),
                "n_members": len(family.get("members", []) or []),
            }
        )
    return rows


def _question_is_auto_assignable(question: Dict[str, Any]) -> bool:
    # Strict family assignment requires all PICO-like dimensions plus objective.
    # Missing information is routed to human review rather than used to create a
    # potentially misleading singleton family.
    for key in ["objective", "population", "intervention", "comparator", "primary_outcome"]:
        value = str(question.get(key, "")).strip().lower()
        if (
            not value
            or value in {"not available", "unknown", "unavailable"}
            or value.startswith("not available from")
        ):
            return False
    return True


def _remove_question_membership(store: Dict[str, Any], article_id: str, question_id: str) -> None:
    for family in store.get("families", {}).values():
        family["members"] = [
            m for m in family.get("members", []) or []
            if not (str(m.get("article_id")) == article_id and str(m.get("question_id")) == question_id)
        ]
    store["pending"] = [
        p for p in store.get("pending", []) or []
        if not (str(p.get("article_id")) == article_id and str(p.get("question_id")) == question_id)
    ]


def _create_family(
    store: Dict[str, Any],
    question: Dict[str, Any],
    *,
    topic_label: Optional[str] = None,
    family_label: Optional[str] = None,
    topic_id: Optional[str] = None,
) -> Dict[str, Any]:
    families = store.setdefault("families", {})
    family_id = _next_numeric_id(families.keys(), "EF")
    if not topic_id:
        topic_ids = {str(f.get("topic_id")) for f in families.values() if f.get("topic_id")}
        topic_id = _next_numeric_id(topic_ids, "ET")
    topic_label = (topic_label or str(question.get("objective", "Evidence topic"))).strip()
    family_label = (family_label or (
        f"{question.get('intervention','Intervention')} vs {question.get('comparator','Comparator')} — "
        f"{question.get('primary_outcome','Outcome')}"
    )).strip()
    family = {
        "family_id": family_id,
        "topic_id": topic_id,
        "topic_label": topic_label,
        "family_label": family_label,
        "signature": {
            "objective": question.get("objective"),
            "population": question.get("population"),
            "intervention": question.get("intervention"),
            "comparator": question.get("comparator"),
            "primary_outcome": question.get("primary_outcome"),
        },
        "members": [],
        "summary": None,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    families[family_id] = family
    return family


def _add_member(family: Dict[str, Any], article_id: str, question: Dict[str, Any]) -> None:
    key = (str(article_id), str(question.get("question_id")))
    members = family.setdefault("members", [])
    if not any((str(m.get("article_id")), str(m.get("question_id"))) == key for m in members):
        members.append(
            {
                "article_id": str(article_id),
                "question_id": str(question.get("question_id")),
                "assigned_at": utc_now(),
            }
        )
    family["updated_at"] = utc_now()
    family["summary"] = None  # member set changed; narrative is stale


def _normalize_compatibility(raw: Any) -> Dict[str, str]:
    dims = ["objective", "population", "intervention", "comparator", "primary_outcome"]
    allowed = {"same", "compatible", "different", "uncertain"}
    src = raw if isinstance(raw, dict) else {}
    return {d: str(src.get(d, "uncertain")) if str(src.get(d, "uncertain")) in allowed else "uncertain" for d in dims}


def _passes_same_family_gate(compat: Dict[str, str]) -> bool:
    # Same-family requires no key dimension to be different or uncertain.
    # Comparator is intentionally included: a different comparator defines a
    # distinct strict family even if it remains in the same broader topic.
    return all(compat.get(d) in {"same", "compatible"} for d in [
        "objective", "population", "intervention", "comparator", "primary_outcome"
    ])


def assign_profile_questions(
    base_dir: Path,
    profile: Dict[str, Any],
    *,
    api_key: str,
    model: str,
    force_reassign: bool = False,
) -> Dict[str, Any]:
    """Assign every eligible evidence question to a strict family.

    Returns counters. Uncertain assignments are stored in `pending` and never
    silently forced into a family.
    """
    store = load_family_store(base_dir)
    article_id = str(profile.get("article_id", ""))
    stats = {"same_family": 0, "new_family": 0, "pending": 0, "skipped": 0}

    if force_reassign:
        # A regenerated profile can change or remove question IDs. Clear every
        # prior membership/pending row for this article before rebuilding its
        # assignments, otherwise stale questions could remain attached to a family.
        for family in store.get("families", {}).values():
            family["members"] = [
                m for m in family.get("members", []) or []
                if str(m.get("article_id")) != article_id
            ]
            family["summary"] = None
        store["pending"] = [
            p for p in store.get("pending", []) or []
            if str(p.get("article_id")) != article_id
        ]

    for question in profile.get("evidence_questions", []) or []:
        qid = str(question.get("question_id", ""))
        if not qid:
            continue

        existing_membership = None
        for family in store.get("families", {}).values():
            if any(str(m.get("article_id")) == article_id and str(m.get("question_id")) == qid for m in family.get("members", []) or []):
                existing_membership = family
                break
        if existing_membership is not None and not force_reassign:
            stats["skipped"] += 1
            continue

        _remove_question_membership(store, article_id, qid)

        if not _question_is_auto_assignable(question):
            store.setdefault("pending", []).append(
                {
                    "article_id": article_id,
                    "question_id": qid,
                    "question": question,
                    "reason": "Objective, intervention, or primary outcome is unavailable; family assignment requires review.",
                    "created_at": utc_now(),
                }
            )
            stats["pending"] += 1
            continue

        families_view = _family_public_view(store)
        if not families_view:
            family = _create_family(store, question)
            _add_member(family, article_id, question)
            stats["new_family"] += 1
            continue

        lm = configure_synthesis_lm(api_key, model)
        predictor = dspy.Predict(EvidenceFamilyAssignmentSignature)
        with dspy.context(lm=lm):
            pred = predictor(
                question_json=json.dumps(question, ensure_ascii=False),
                existing_families_json=json.dumps(families_view, ensure_ascii=False),
            )
        raw = extract_json_object(pred.output_json)
        decision = str(raw.get("decision", "needs_review"))
        compat = _normalize_compatibility(raw.get("compatibility"))
        family_id = str(raw.get("family_id") or "")
        related_family_id = str(raw.get("related_family_id") or family_id or "")
        reason = str(raw.get("reason", ""))

        if decision == "same_family" and family_id in store.get("families", {}) and _passes_same_family_gate(compat):
            _add_member(store["families"][family_id], article_id, question)
            stats["same_family"] += 1
        elif decision == "related_but_distinct" and related_family_id in store.get("families", {}):
            related = store["families"][related_family_id]
            family = _create_family(
                store,
                question,
                topic_id=related.get("topic_id"),
                topic_label=related.get("topic_label"),
                family_label=str(raw.get("family_label") or "").strip() or None,
            )
            _add_member(family, article_id, question)
            stats["new_family"] += 1
        elif decision == "no_related_family":
            family = _create_family(
                store,
                question,
                topic_label=str(raw.get("topic_label") or "").strip() or None,
                family_label=str(raw.get("family_label") or "").strip() or None,
            )
            _add_member(family, article_id, question)
            stats["new_family"] += 1
        else:
            store.setdefault("pending", []).append(
                {
                    "article_id": article_id,
                    "question_id": qid,
                    "question": question,
                    "candidate_family_id": family_id or related_family_id or None,
                    "compatibility": compat,
                    "reason": reason or "Family compatibility was uncertain or failed the strict membership gate.",
                    "proposed_topic_label": raw.get("topic_label"),
                    "proposed_family_label": raw.get("family_label"),
                    "created_at": utc_now(),
                }
            )
            stats["pending"] += 1

    # Remove strict families that became empty after profile regeneration.
    store["families"] = {
        fid: fam for fid, fam in store.get("families", {}).items()
        if fam.get("members")
    }
    save_family_store(base_dir, store)
    return stats


def resolve_pending_assignment(
    base_dir: Path,
    *,
    article_id: str,
    question_id: str,
    action: str,
    family_id: Optional[str] = None,
) -> None:
    """Human resolution for an ambiguous family assignment.

    action = existing_family | new_family
    """
    store = load_family_store(base_dir)
    pending = None
    for p in store.get("pending", []) or []:
        if str(p.get("article_id")) == str(article_id) and str(p.get("question_id")) == str(question_id):
            pending = p
            break
    if pending is None:
        raise KeyError("Pending assignment not found")
    question = pending.get("question", {})
    _remove_question_membership(store, str(article_id), str(question_id))

    if action == "existing_family":
        if not family_id or family_id not in store.get("families", {}):
            raise ValueError("Choose a valid existing family")
        _add_member(store["families"][family_id], str(article_id), question)
    elif action == "new_family":
        related_id = str(pending.get("candidate_family_id") or "")
        if related_id in store.get("families", {}):
            related = store["families"][related_id]
            family = _create_family(
                store,
                question,
                topic_id=related.get("topic_id"),
                topic_label=related.get("topic_label"),
                family_label=pending.get("proposed_family_label"),
            )
        else:
            family = _create_family(
                store,
                question,
                topic_label=pending.get("proposed_topic_label"),
                family_label=pending.get("proposed_family_label"),
            )
        _add_member(family, str(article_id), question)
    else:
        raise ValueError("Unknown pending-assignment action")
    save_family_store(base_dir, store)


def get_family_member_context(
    family: Dict[str, Any],
    profiles: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for member in family.get("members", []) or []:
        aid = str(member.get("article_id", ""))
        qid = str(member.get("question_id", ""))
        profile = profiles.get(aid)
        if not profile:
            continue
        question = next((q for q in profile.get("evidence_questions", []) or [] if str(q.get("question_id")) == qid), None)
        rows.append({"article_id": aid, "question_id": qid, "question": question, "profile": profile})
    return rows


def generate_family_summary(
    base_dir: Path,
    family_id: str,
    *,
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    store = load_family_store(base_dir)
    family = store.get("families", {}).get(family_id)
    if not family:
        raise KeyError(f"Unknown family: {family_id}")
    profiles = load_all_profiles(base_dir)
    member_context = get_family_member_context(family, profiles)
    lm = configure_synthesis_lm(api_key, model)
    predictor = dspy.Predict(FamilyNarrativeSignature)
    family_view = {
        "family_id": family.get("family_id"),
        "topic_label": family.get("topic_label"),
        "family_label": family.get("family_label"),
        "signature": family.get("signature"),
    }
    compact_profiles = []
    for row in member_context:
        p = row["profile"]
        compact_profiles.append(
            {
                "article_id": row["article_id"],
                "question": row["question"],
                "sample_size": p.get("sample_size"),
                "population": p.get("population"),
                "duration_follow_up": p.get("duration_follow_up"),
                "effect_result": p.get("effect_result"),
                "safety": p.get("safety"),
                "consort_completeness": p.get("consort_completeness"),
            }
        )
    with dspy.context(lm=lm):
        pred = predictor(
            family_json=json.dumps(family_view, ensure_ascii=False),
            member_profiles_json=json.dumps(compact_profiles, ensure_ascii=False),
        )
    summary = extract_json_object(pred.output_json)
    family["summary"] = {
        "summary": str(summary.get("summary", "")),
        "heterogeneity": [str(x) for x in summary.get("heterogeneity", []) or []],
        "evidence_gaps": [str(x) for x in summary.get("evidence_gaps", []) or []],
        "generated_at": utc_now(),
    }
    save_family_store(base_dir, store)
    return family["summary"]


# -----------------------------------------------------------------------------
# UI-friendly flattening
# -----------------------------------------------------------------------------
def field_value(profile: Dict[str, Any], key: str) -> str:
    obj = profile.get(key, {})
    if isinstance(obj, dict):
        return str(obj.get("value") or "Not available")
    return str(obj or "Not available")


def family_table_rows(
    family: Dict[str, Any],
    profiles: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for member in family.get("members", []) or []:
        aid = str(member.get("article_id", ""))
        qid = str(member.get("question_id", ""))
        p = profiles.get(aid)
        if not p:
            continue
        q = next((x for x in p.get("evidence_questions", []) or [] if str(x.get("question_id")) == qid), {})
        comp = p.get("consort_completeness", {}) or {}
        rows.append(
            {
                "Study": p.get("study_label") or aid,
                "Study ID": aid,
                "Sample Size": (p.get("sample_size", {}) or {}).get("display", "Not available"),
                "Population": q.get("population") or field_value(p, "population"),
                "Intervention": q.get("intervention") or field_value(p, "intervention"),
                "Comparator": q.get("comparator") or field_value(p, "comparator"),
                "Duration / Follow-up": field_value(p, "duration_follow_up"),
                "Primary Outcome": q.get("primary_outcome") or field_value(p, "primary_outcome"),
                "Effect / Result": field_value(p, "effect_result"),
                "Safety": field_value(p, "safety"),
                "CONSORT Completeness": comp.get("display", "Not available"),
            }
        )
    return rows


def topic_table_rows(
    family_store: Dict[str, Any],
    topic_id: str,
    profiles: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """All studies sharing a broad topic, spanning every strict family under it.

    Includes a "Family" column since studies in the same topic may still
    differ on dose/route/comparator and not be directly comparable.
    """
    rows: List[Dict[str, Any]] = []
    for family in (family_store.get("families", {}) or {}).values():
        if str(family.get("topic_id", "")) != str(topic_id):
            continue
        for row in family_table_rows(family, profiles):
            row["Family"] = family.get("family_label") or family.get("family_id")
            rows.append(row)
    return rows
