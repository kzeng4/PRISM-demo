"""CONSORT live inference wrapper (v4).

Pipeline adapted from prompt_optimize_0820_section_labeling_v26_boundary_hardened.ipynb.

Online path:
    .txt -> deterministic paragraph/sentence hierarchy -> global section router
         -> article-level cross-reference memory -> section-specific CONSORT extractors
         -> deterministic validation/deduplication -> demo record

Deliberately NOT included in v4:
    cross-section verifier, candidate adjudicator, verifier recovery loop,
    targeted router recovery, GEPA optimization, or any gold-label/evaluation logic.

The returned record is intentionally compatible with the precomputed
``pre_gepa_test_demo_records.json`` structure used by the Streamlit HITL app.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import dspy
except Exception:  # pragma: no cover - runtime dependency is optional for preprocessing
    dspy = None


# -----------------------------------------------------------------------------
# v26 section / item configuration
# -----------------------------------------------------------------------------
CANONICAL_SECTIONS = [
    "title_abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "other_information",
]

SECTION_DESCRIPTION = """
Allowed canonical sections:
- title_abstract: title, abstract, structured summary, abstract-like trial summary
- introduction: scientific background, rationale, objectives, hypotheses
- methods: trial design, participants, interventions, outcomes, randomization, blinding, sample size, statistical methods, protocol/method procedures
- results: participant flow, recruitment/follow-up dates when reported as results, baseline data, numbers analyzed, outcome results, ancillary analyses, harms/adverse events
- discussion: limitations, generalizability/applicability, interpretation, implications, conclusions
- other_information: trial registration, protocol availability, funding, conflicts of interest, acknowledgements, author contributions, ethics statements, data availability, references, miscellaneous article metadata
""".strip()

SECTION_TO_ITEMS: Dict[str, List[str]] = {
    "title_abstract": ["1a", "1b"],
    "introduction": ["2a", "2b"],
    "methods": [
        "3a", "3b", "4a", "4b", "5", "6a", "6b", "7a", "7b",
        "8a", "8b", "9", "10", "11a", "11b", "12a", "12b",
    ],
    "results": ["13a", "13b", "14a", "14b", "15", "16", "17a", "17b", "18", "19"],
    "discussion": ["20", "21", "22"],
    "other_information": ["23", "24", "25"],
}

# Matches v26: title/abstract extraction is currently disabled.
ACTIVE_EXTRACTION_SECTIONS = [
    "introduction",
    "methods",
    "results",
    "discussion",
    "other_information",
]
SECTION_ALLOWED_ITEMS = {s: list(SECTION_TO_ITEMS[s]) for s in ACTIVE_EXTRACTION_SECTIONS}

ITEM_ORDER = [
    "1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b", "5", "6a", "6b",
    "7a", "7b", "8a", "8b", "9", "10", "11a", "11b", "12a", "12b",
    "13a", "13b", "14a", "14b", "15", "16", "17a", "17b", "18", "19",
    "20", "21", "22", "23", "24", "25",
]

ITEM_TO_SECTION = {
    item: section
    for section, items in SECTION_TO_ITEMS.items()
    for item in items
}

ARTICLE_CONTEXT_KEYS = [
    "trial_structure", "participants_setting", "interventions", "outcomes", "sample_size",
    "randomization", "blinding", "analysis", "timeline_flow", "registration_protocol",
]

MAX_WORKED_EXAMPLES = 5
ROUTER_MISSING_FALLBACK_ALL_ACTIVE = True


@dataclass
class PipelineConfig:
    """Runtime configuration for live v4 inference."""

    model: str = "openai/gpt-5.6-terra"
    temperature: float = 1.0
    max_tokens: int = 32000
    max_retries: int = 10
    reasoning_effort: str = "medium"
    enable_article_context: bool = True

    # v26 assets. The prototype is required for faithful v26 extraction prompts.
    prototype_path: Optional[str] = None
    annotation_guideline_path: Optional[str] = None
    optimized_router_path: Optional[str] = None

    # If True, missing prototype is an error. Guideline has the same fallback behavior
    # as v26, but the app exposes a warning when it is absent.
    strict_prototype: bool = True


# -----------------------------------------------------------------------------
# Deterministic TXT preprocessing
# -----------------------------------------------------------------------------
_ABBREVIATIONS = [
    "e.g.", "i.e.", "et al.", "vs.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.",
    "Fig.", "Figs.", "Eq.", "Eqs.", "No.", "Nos.", "Inc.", "Ltd.", "St.",
    "approx.", "ref.", "refs.", "vol.", "pp.", "al.",
]
_DOT_TOKEN = "<CONSORT_DOT>"


def normalize_model_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", text).strip()


def _protect_sentence_internal_periods(text: str) -> str:
    out = text
    # Decimal points and numeric version-like tokens.
    out = re.sub(r"(?<=\d)\.(?=\d)", _DOT_TOKEN, out)

    # Common abbreviations, case-insensitive.
    for abbr in sorted(_ABBREVIATIONS, key=len, reverse=True):
        pat = re.compile(re.escape(abbr), re.IGNORECASE)
        out = pat.sub(lambda m: m.group(0).replace(".", _DOT_TOKEN), out)

    # Initials such as J. Smith or A. B. Smith. Protect when a period is followed by
    # an uppercase token; this avoids obvious author/initial splits.
    out = re.sub(r"\b([A-Z])\.(?=\s+[A-Z])", lambda m: m.group(1) + _DOT_TOKEN, out)

    # Time-like a.m./p.m.
    out = re.sub(
        r"\b([ap])\.m\.",
        lambda m: m.group(1) + _DOT_TOKEN + "m" + _DOT_TOKEN,
        out,
        flags=re.IGNORECASE,
    )
    return out


def split_scientific_sentences(paragraph: str) -> List[str]:
    """Conservative dependency-free sentence splitter for RCT prose.

    The v26 notebook used a regex fallback. This version keeps that logic but adds
    protection for decimals and common scientific abbreviations, which matters for
    user-uploaded raw text. It never changes paragraph boundaries.
    """
    paragraph = normalize_model_text(paragraph)
    if not paragraph:
        return []

    protected = _protect_sentence_internal_periods(paragraph)
    pieces = re.split(
        r"(?<=[.!?])\s+(?=(?:[\"'“”‘’\(\[])?[A-Z0-9])",
        protected,
    )
    result = []
    for piece in pieces:
        piece = piece.replace(_DOT_TOKEN, ".").strip()
        if piece:
            result.append(piece)
    return result or [paragraph]


def split_txt_paragraphs(text: str) -> List[str]:
    """Split raw TXT on blank lines, preserving source order.

    Wrapped lines inside a paragraph are merged with one space. This is appropriate
    for the sample clean-text format supplied for v4 and keeps headings/captions as
    standalone paragraphs when they are separated by blank lines.
    """
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    chunks = re.split(r"\n\s*\n+", text.strip())
    paragraphs = []
    for chunk in chunks:
        lines = [line.strip() for line in chunk.split("\n") if line.strip()]
        para = normalize_model_text(" ".join(lines))
        if para:
            paragraphs.append(para)
    return paragraphs


def preprocess_txt(text: str, article_id: str) -> Dict[str, Any]:
    """Create the canonical live-input hierarchy and stable source identifiers.

    ID contract for raw TXT (there is no pre-existing source PID):
      * pid  = 1-based paragraph position in the uploaded TXT, stored as a string
      * pnum = 0-based paragraph position; therefore pnum == int(pid) - 1
      * sid  = global 1-based sentence position in reading order: S1, S2, ...

    IDs are deterministic for an unchanged TXT file. They should be treated as tied
    to this exact preprocessing version/input; editing blank-line structure can change IDs.
    """
    article_id = str(article_id or "uploaded_article").strip()
    paragraphs_text = split_txt_paragraphs(text)
    if not paragraphs_text:
        raise ValueError("The uploaded TXT file contains no non-empty paragraphs.")

    paragraphs: List[Dict[str, Any]] = []
    flat_sentences: List[Dict[str, Any]] = []
    sid_counter = 1

    for pnum, para_text in enumerate(paragraphs_text):
        pid = str(pnum + 1)
        sentence_texts = split_scientific_sentences(para_text)
        sentence_rows = []
        for sent_text in sentence_texts:
            sid = f"S{sid_counter}"
            sid_counter += 1
            row = {
                "sid": sid,
                "pnum": pnum,
                "pid": pid,
                "text": sent_text,
                "model_text": normalize_model_text(sent_text),
            }
            sentence_rows.append(row)
            flat_sentences.append(dict(row))

        paragraphs.append({
            "pnum": pnum,
            "pid": pid,
            "text": para_text,
            "model_text": normalize_model_text(para_text),
            "predicted_sections": [],
            "sentences": sentence_rows,
        })

    model_hierarchy = [
        {
            "pnum": p["pnum"],
            "paragraph_text": p["model_text"],
            "sentences": [
                {"sid": s["sid"], "text": s["model_text"]}
                for s in p["sentences"]
            ],
        }
        for p in paragraphs
    ]

    content_hash = hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()[:16]
    return {
        "article_id": article_id,
        "content_hash": content_hash,
        "id_contract": {
            "pid": "1-based paragraph position in this uploaded TXT; stringified for JSON; never inferred from sid.",
            "pnum": "0-based paragraph position in this uploaded TXT; pnum == int(pid)-1.",
            "sid": "Global 1-based sentence position in reading order (S1, S2, ...); copied verbatim downstream and never regenerated by the LLM.",
            "stability": "Stable for the identical TXT under consort_live_pipeline_v4 preprocessing. Editing blank-line paragraph boundaries can change downstream IDs.",
        },
        "paragraphs": paragraphs,
        "sentences": flat_sentences,
        "article_input_json": json.dumps(model_hierarchy, ensure_ascii=False),
    }


def preprocess_txt_file(path: str, article_id: Optional[str] = None) -> Dict[str, Any]:
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8", errors="replace")
    return preprocess_txt(text, article_id or path_obj.stem)


# -----------------------------------------------------------------------------
# Shared JSON / section helpers ported from v26
# -----------------------------------------------------------------------------
def _item_sort_key(x: Any) -> Tuple[int, str]:
    m = re.match(r"^(\d+)([a-z]?)$", str(x))
    if not m:
        return (999, str(x))
    return (int(m.group(1)), m.group(2))


def normalize_section_name(raw: Any) -> str:
    if raw is None:
        return "other_information"
    s = str(raw).strip().lower()
    s = s.strip("[](){}'\" ")
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    if not s:
        return "other_information"

    canonical_guess = s.replace(" ", "_")
    if canonical_guess in CANONICAL_SECTIONS:
        return canonical_guess

    if any(k in s for k in ["title", "abstract", "summary", "synopsis"]):
        return "title_abstract"
    if any(k in s for k in ["introduction", "background", "rationale", "objective", "objectives", "hypothesis", "hypotheses"]):
        return "introduction"
    method_terms = [
        "method", "methods", "material", "materials", "design", "participant", "participants",
        "patient", "patients", "intervention", "outcome", "random", "randomisation",
        "randomization", "blinding", "masking", "sample size", "statistical", "statistics",
        "analysis", "protocol", "eligibility", "allocation", "procedure",
    ]
    if any(k in s for k in method_terms):
        return "methods"
    result_terms = [
        "result", "results", "finding", "findings", "participant flow", "baseline",
        "numbers analysed", "numbers analyzed", "estimation", "harms", "adverse",
        "recruitment", "follow-up", "ancillary", "subgroup", "table 1",
    ]
    if any(k in s for k in result_terms):
        return "results"
    if any(k in s for k in [
        "discussion", "limitation", "limitations", "generalizability", "generalisability",
        "applicability", "interpretation", "conclusion", "conclusions", "implication", "implications",
    ]):
        return "discussion"
    if any(k in s for k in [
        "funding", "registration", "trial registration", "conflict", "interest", "acknowledg",
        "ethic", "author", "contribution", "data availability", "reference", "appendix",
        "supplement", "protocol availability",
    ]):
        return "other_information"
    return "other_information"


def _try_parse_list_string(value: Any) -> Optional[List[Any]]:
    if not isinstance(value, str):
        return None
    v = value.strip()
    if (v.startswith("[") and v.endswith("]")) or (v.startswith("(") and v.endswith(")")):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        except Exception:
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return None
    return None


def clean_section_list(value: Any) -> List[str]:
    parsed = _try_parse_list_string(value)
    if parsed is not None:
        value = parsed
    if isinstance(value, str):
        v = value.strip()
        if any(sep in v for sep in [";", "|", ","]):
            value = [p for p in re.split(r"\s*[;,|]\s*", v) if p]
        else:
            value = [v]
    if not isinstance(value, (list, tuple, set)):
        value = [value]

    cleaned = []
    for x in value:
        sec = normalize_section_name(x)
        if sec in CANONICAL_SECTIONS and sec not in cleaned:
            cleaned.append(sec)
    return cleaned or ["other_information"]


def extract_json_object(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    text = str(text or "").strip()
    if not text:
        raise ValueError("Empty output")
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
        obj = json.loads(text[start:end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Could not find valid JSON object")


def parse_article_paragraphs_json(raw: Any) -> Dict[int, Dict[str, Any]]:
    rows = json.loads(raw) if isinstance(raw, str) else raw
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        pnum = int(r["pnum"])
        out[pnum] = {
            "pnum": pnum,
            "paragraph_text": str(r.get("paragraph_text", r.get("text", ""))),
            "sentences": [
                {"sid": str(s.get("sid", "")), "text": str(s.get("text", ""))}
                for s in r.get("sentences", [])
            ],
        }
    return out


def sid_text_lookup_from_hierarchy(article_paragraphs_json: Any) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for pnum, p in parse_article_paragraphs_json(article_paragraphs_json).items():
        for s in p.get("sentences", []):
            lookup[str(s["sid"])] = {"pnum": pnum, "text": s.get("text", "")}
    return lookup


def validate_router_prediction(
    article_input_json: str,
    section_output_json: Any,
    active_sections: Optional[Sequence[str]] = None,
) -> Tuple[Dict[int, List[str]], Dict[str, Any]]:
    active_sections = list(active_sections or ACTIVE_EXTRACTION_SECTIONS)
    paragraphs = parse_article_paragraphs_json(article_input_json)
    expected = set(paragraphs)
    diagnostics = {"duplicates": [], "missing": [], "unknown_labels": [], "parse_error": None}
    mapping: Dict[int, set] = {p: set() for p in expected}

    try:
        obj = extract_json_object(section_output_json)
        seen = set()
        for row in obj.get("paragraph_sections", []):
            try:
                pnum = int(row.get("pnum"))
            except Exception:
                continue
            if pnum not in expected:
                continue
            if pnum in seen:
                diagnostics["duplicates"].append(pnum)
            seen.add(pnum)
            raw_sections = row.get("sections", [])
            cleaned = clean_section_list(raw_sections)
            raw_iter = raw_sections if isinstance(raw_sections, list) else [raw_sections]
            unknown = [
                str(x) for x in raw_iter
                if str(x).strip().replace(" ", "_").lower() not in CANONICAL_SECTIONS
                and normalize_section_name(x) not in CANONICAL_SECTIONS
            ]
            diagnostics["unknown_labels"].extend(unknown)
            mapping[pnum].update(s for s in cleaned if s in active_sections or s in CANONICAL_SECTIONS)
    except Exception as e:
        diagnostics["parse_error"] = repr(e)

    missing = sorted(p for p in expected if not mapping[p])
    diagnostics["missing"] = missing
    if ROUTER_MISSING_FALLBACK_ALL_ACTIVE:
        for pnum in missing:
            mapping[pnum].update(active_sections)
    return {p: sorted(v, key=lambda s: CANONICAL_SECTIONS.index(s)) for p, v in mapping.items()}, diagnostics


def selected_paragraphs_for_target_section(
    article_paragraphs_json: str,
    pred_sections_map: Dict[int, List[str]],
    target_section: str,
) -> str:
    target = normalize_section_name(target_section)
    paragraphs = parse_article_paragraphs_json(article_paragraphs_json)
    selected = [
        p for pnum, p in sorted(paragraphs.items())
        if target in set(pred_sections_map.get(pnum, []))
    ]
    return json.dumps(selected, ensure_ascii=False)


def sanitize_article_context_json(raw_context: Any, article_input_json: str) -> Tuple[str, List[Dict[str, Any]]]:
    if not raw_context:
        return "", []
    valid_sids = set(sid_text_lookup_from_hierarchy(article_input_json))
    events: List[Dict[str, Any]] = []
    try:
        obj = extract_json_object(raw_context)
    except Exception as e:
        return "", [{"kind": "context_parse_error", "detail": repr(e)}]

    for k in [k for k in obj if k not in ARTICLE_CONTEXT_KEYS]:
        events.append({"kind": "context_unknown_category", "category": k})
    obj = {k: obj.get(k, []) for k in ARTICLE_CONTEXT_KEYS}

    for category, facts in list(obj.items()):
        if not isinstance(facts, list):
            obj[category] = []
            events.append({"kind": "context_bad_category", "category": category})
            continue
        cleaned_facts = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            fact = dict(fact)
            sids = [str(s) for s in fact.get("source_sids", []) if str(s)]
            invalid = [s for s in sids if s not in valid_sids]
            if invalid:
                events.append({"kind": "context_invalid_sid", "category": category, "sids": invalid})
            fact["source_sids"] = [s for s in sids if s in valid_sids]
            if fact.get("certainty") not in {"high", "medium", "low"}:
                fact["certainty"] = "low"
            cleaned_facts.append(fact)
        obj[category] = cleaned_facts
    return json.dumps(obj, ensure_ascii=False), events


# -----------------------------------------------------------------------------
# v26 prototype / annotation-guideline prompt construction
# -----------------------------------------------------------------------------
def load_consort_prototype(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        proto = json.load(f)
    for key in ["metadata", "item_prototypes", "pairwise_boundary_rules", "global_rules"]:
        if key not in proto:
            raise ValueError(f"Prototype JSON missing key: {key}")
    return proto


def load_annotation_guideline(path: Optional[str]) -> Dict[str, Any]:
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Exact fallback used in v26.
    return {
        "general_annotation_rules": [
            "Annotation guideline file was not found at runtime.",
            "Use the CONSORT checklist definitions, item prototypes, global rules, and pairwise boundary rules as the active annotation guidance.",
            "Do not invent evidence. Extract exact sentence IDs from the provided sentence hierarchy only.",
        ],
        "item_specific_rules": {},
    }


def section_prototype_snapshot(proto: Dict[str, Any], section_name: str, allowed_items: Sequence[str]) -> Dict[str, Any]:
    allowed = {str(x) for x in allowed_items}
    item_prototypes = {
        item: copy.deepcopy(proto.get("item_prototypes", {}).get(item, {}))
        for item in sorted(allowed, key=str)
        if item in proto.get("item_prototypes", {})
    }
    pairwise_rules = {}
    for rule_id, rule in proto.get("pairwise_boundary_rules", {}).items():
        pair = {str(x) for x in rule.get("items", [])}
        if pair and pair.issubset(allowed):
            pairwise_rules[rule_id] = copy.deepcopy(rule)
    worked_examples = [
        copy.deepcopy(ex)
        for ex in proto.get("worked_examples", {}).get(section_name, [])
    ][:MAX_WORKED_EXAMPLES]
    return {
        "section": section_name,
        "allowed_items": sorted(allowed, key=str),
        "item_prototypes": item_prototypes,
        "pairwise_boundary_rules": pairwise_rules,
        "section_strategy": [],
        "worked_examples": worked_examples,
    }


def render_item_definitions(items: Sequence[str], proto: Dict[str, Any]) -> str:
    lines = []
    for item in sorted(items, key=_item_sort_key):
        ip = proto.get("item_prototypes", {}).get(item, {})
        lines.append(f"- {item}: {ip.get('definition', '')}")
    return "\n".join(lines)


def render_global_rules_text(proto: Dict[str, Any], max_rules: int = 18) -> str:
    return "\n".join(f"- {r}" for r in proto.get("global_rules", [])[:max_rules])


def render_section_guideline_text(guideline_obj: Dict[str, Any], allowed_items: Sequence[str]) -> str:
    allowed = sorted({str(x) for x in allowed_items}, key=_item_sort_key)
    item_rules = guideline_obj.get("item_specific_rules", {})
    lines = ["General annotation rules:"]
    for rule in guideline_obj.get("general_annotation_rules", []):
        lines.append(f"- {rule}")
    lines.extend(["", "Item-specific annotation rules:"])
    for item in allowed:
        rule = item_rules.get(item)
        if not rule:
            continue
        parts = []
        if rule.get("title"):
            parts.append(rule["title"])
        if rule.get("special_rule"):
            parts.append(f"Special rule: {rule['special_rule']}")
        for guidance in rule.get("cross_item_guidance", []):
            parts.append(f"Cross-item guidance: {guidance}")
        if parts:
            lines.append(f"- {item}: " + " | ".join(parts))
    return "\n".join(lines)


def render_fixed_section_extraction_spec(
    section_name: str,
    allowed_items: Sequence[str],
    proto: Dict[str, Any],
    guideline_obj: Dict[str, Any],
) -> str:
    official_defs = render_item_definitions(allowed_items, proto)
    global_rules = render_global_rules_text(proto)
    guideline_excerpt = render_section_guideline_text(guideline_obj, allowed_items)
    return f"""
EXTRACTOR SPECIFICATION

Task:
Extract CONSORT item/sentence-ID pairs from sentence hierarchies routed to the {section_name} extractor.

Extractor section:
{section_name}

Allowed item IDs:
{json.dumps(list(allowed_items))}

Official definitions:
{official_defs}

Input:
selected_paragraphs_json is a JSON array of paragraph objects. Each paragraph contains pnum,
paragraph_text, and sentence objects with sid and text.

Required output JSON:
{{
  "article_id": "<article id>",
  "section": "{section_name}",
  "extractions": [
    {{
      "item": "<allowed CONSORT item id>",
      "pnum": 3,
      "sid": "<sentence id copied exactly from input>",
      "score": 0.0
    }}
  ]
}}

Output and extraction constraints:
- Return valid JSON only, with no markdown or commentary.
- Output sid, never full sentence text.
- Copy sid exactly from the input; never invent or normalize it.
- Output only the allowed item IDs listed above.
- Do not infer missing evidence or use hidden gold labels.
- Do not annotate bare section headers.
- A sentence may support multiple items only when it explicitly satisfies each item.
- If no evidence exists, return an empty extractions list.
- score must be numeric and between 0 and 1.

Annotation guideline:
{guideline_excerpt}

Global extraction rules:
{global_rules}
""".strip()


def render_mutable_section_prototype_memory(
    section_name: str,
    allowed_items: Sequence[str],
    proto: Dict[str, Any],
) -> str:
    snapshot = section_prototype_snapshot(proto, section_name, allowed_items)
    return (
        "SECTION-SPECIFIC MUTABLE PROTOTYPE MEMORY\n"
        "This is the only extractor component that may be revised during optimization.\n"
        f"Scope is strictly section={section_name} and items={json.dumps(list(allowed_items))}.\n"
        "Revise only item_prototypes, pairwise_boundary_rules, section_strategy, or worked_examples.\n"
        f"worked_examples is capped at {MAX_WORKED_EXAMPLES} entries; each entry must have "
        '"snippet" (short verbatim-style text span), "correct_items" (subset of allowed_items), '
        'and "rationale" (why those items and not others). Use it for concrete illustrative cases, '
        "not as a place to restate rules already covered by item_prototypes.\n"
        "Do not introduce item IDs outside allowed_items and do not restate or rewrite the fixed task specification.\n\n"
        + json.dumps(snapshot, ensure_ascii=False, indent=2)
    )


# -----------------------------------------------------------------------------
# DSPy modules adapted directly from v26
# -----------------------------------------------------------------------------
SECTION_ROUTER_FIXED_TASK_SPEC = f"""
Classify each paragraph of a CONSORT randomized trial article into one or more canonical sections.

You must use exactly these canonical labels:
{", ".join(CANONICAL_SECTIONS)}.

INPUT FORMAT:
The article input may be either:
1) a clean paragraph-numbered text format, or
2) a JSON array of paragraph objects with this structure:
   [
     {{
       "pnum": 0,
       "paragraph_text": "...",
       "sentences": [{{"sid": "...", "text": "..."}}]
     }}
   ]
In the JSON format, use pnum as the paragraph identifier. Sentence IDs are provided for later extraction;
section classification is still paragraph-level.

This is a multi-label classification task:
- Most paragraphs should have one section label.
- Some paragraphs may need multiple labels when they contain content belonging to multiple CONSORT sections.
- Use multiple labels only when the paragraph explicitly contains multiple section functions.

{SECTION_DESCRIPTION}

Hard output rules:
- Return valid JSON only. No markdown and no explanation.
- Include every paragraph number present in the article input exactly once.
- Do not repeat paragraph text in the output.
- Each paragraph must have a non-empty "sections" list.
- Use the article order as global context, but classify by the paragraph's actual content.

Output format:
{{
  "article_id": "<article_id>",
  "paragraph_sections": [
    {{"pnum": 0, "sections": ["title_abstract"]}},
    {{"pnum": 1, "sections": ["introduction"]}},
    {{"pnum": 2, "sections": ["methods", "results"]}}
  ]
}}
""".strip()

ARTICLE_CONTEXT_FIXED_TASK_SPEC = """
Build a compact CONSORT-oriented cross-reference memory from the full article.
Return JSON with exactly these keys:
{
  "trial_structure": [fact, ...],
  "participants_setting": [fact, ...],
  "interventions": [fact, ...],
  "outcomes": [fact, ...],
  "sample_size": [fact, ...],
  "randomization": [fact, ...],
  "blinding": [fact, ...],
  "analysis": [fact, ...],
  "timeline_flow": [fact, ...],
  "registration_protocol": [fact, ...]
}
Each fact must be:
{"value": "short factual phrase", "source_sids": ["exact sid", ...],
 "certainty": "high|medium|low"}

Rules:
- Use only facts explicitly supported by the article; do not infer unstated details.
- source_sids must be copied from the article input and should be the smallest useful support set.
- Keep facts short and cross-reference oriented (e.g., primary outcome identity/timepoint,
  arm mapping, analysis population, planned N, randomization/blinding structure).
- Empty categories are []. Return JSON only.
""".strip()


if dspy is not None:
    class ArticleSectionLabeling(dspy.Signature):
        """
        Decide section membership for each paragraph using the fixed task specification.

        Routing heuristics for ambiguous cases:
        - Do not classify tables, figures, or captions as other_information solely because they are tables/figures.
          Classify table/figure captions by content:
          * trial profile, study flow, participant flow, baseline characteristics, outcomes, Kaplan-Meier curves, adverse events, endpoints, efficacy, or safety -> results
          * intervention details, eligibility, randomization, blinding, assessment procedures -> methods
          * funding, competing interests, ethics approval, author contributions, acknowledgments, provenance, protocol availability -> other_information
        - If a paragraph contains CONSORT item evidence from multiple item-native sections, include all relevant sections.
        """
        article_id = dspy.InputField(desc="Article identifier")
        fixed_task_spec = dspy.InputField(desc="Frozen router task specification")
        article_text = dspy.InputField(desc="Canonical article paragraph/sentence hierarchy")
        output_json = dspy.OutputField(desc="Valid JSON object with article_id and paragraph_sections")

    class ArticleSectionLabeler(dspy.Module):
        def __init__(self):
            super().__init__()
            self.label = dspy.Predict(ArticleSectionLabeling)

        def forward(self, article_id: str, article_text: str):
            return self.label(
                article_id=str(article_id),
                fixed_task_spec=SECTION_ROUTER_FIXED_TASK_SPEC,
                article_text=article_text,
            )

    class ArticleContextMemory(dspy.Signature):
        article_id = dspy.InputField(desc="Article identifier")
        fixed_task_spec = dspy.InputField(desc="Frozen cross-section memory schema and grounding rules")
        article_text = dspy.InputField(desc="Canonical article paragraph/sentence hierarchy")
        output_json = dspy.OutputField(desc="Grounded cross-section memory JSON")

    class ArticleAnchorExtractor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.extract = dspy.Predict(ArticleContextMemory)

        def forward(self, article_id: str, article_text: str):
            return self.extract(
                article_id=str(article_id),
                fixed_task_spec=ARTICLE_CONTEXT_FIXED_TASK_SPEC,
                article_text=article_text,
            )

    def make_section_extraction_signature(section_name: str, mutable_prototype_text: str):
        try:
            return dspy.Signature(
                "article_id, fixed_task_spec, article_context_json, retry_feedback, selected_paragraphs_json -> output_json",
                instructions=mutable_prototype_text,
            )
        except Exception:
            class SectionExtraction(dspy.Signature):
                article_id = dspy.InputField()
                fixed_task_spec = dspy.InputField()
                article_context_json = dspy.InputField()
                retry_feedback = dspy.InputField()
                selected_paragraphs_json = dspy.InputField()
                output_json = dspy.OutputField()
            SectionExtraction.__doc__ = mutable_prototype_text
            return SectionExtraction

    class GenericSectionExtractor(dspy.Module):
        def __init__(self, section_name: str, mutable_prototype_text: str, fixed_task_spec: str):
            super().__init__()
            self.section_name = section_name
            self.fixed_task_spec = fixed_task_spec
            self.extract = dspy.Predict(
                make_section_extraction_signature(section_name, mutable_prototype_text)
            )

        def forward(
            self,
            article_id: str,
            selected_paragraphs_json: str,
            article_context_json: str = "",
            retry_feedback: str = "",
        ):
            return self.extract(
                article_id=article_id,
                fixed_task_spec=self.fixed_task_spec,
                article_context_json=article_context_json,
                retry_feedback=retry_feedback,
                selected_paragraphs_json=selected_paragraphs_json,
            )
else:  # pragma: no cover
    ArticleSectionLabeler = None
    ArticleAnchorExtractor = None
    GenericSectionExtractor = None


def _configure_dspy(api_key: str, config: PipelineConfig) -> "dspy.LM":
    """Build a DSPy LM for the live pipeline.

    Returns the LM instead of calling `dspy.configure(lm=...)` because DSPy's
    global configure is locked to whichever thread calls it first, and
    Streamlit reruns the script on a new thread per interaction. Callers
    should run the pipeline inside `with dspy.context(lm=...)`, which DSPy
    allows from any thread.
    """
    if dspy is None:
        raise ImportError("DSPy is not installed. Install it before running live extraction (e.g. `pip install dspy`).")
    if not str(api_key or "").strip():
        raise ValueError("An OpenAI API key is required for live section routing/extraction.")

    kwargs: Dict[str, Any] = {
        "temperature": config.temperature,
        "api_key": str(api_key).strip(),
        "max_retries": config.max_retries,
        "max_tokens": config.max_tokens,
    }
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort

    try:
        lm = dspy.LM(config.model, **kwargs)
    except TypeError:
        # Compatibility with DSPy/provider combinations that do not expose reasoning_effort.
        kwargs.pop("reasoning_effort", None)
        lm = dspy.LM(config.model, **kwargs)
    return lm


def _make_router(config: PipelineConfig):
    if dspy is None:
        raise ImportError("DSPy is not installed.")
    router = ArticleSectionLabeler()
    if config.optimized_router_path:
        path = Path(config.optimized_router_path)
        if path.exists():
            router.load(str(path))
        else:
            raise FileNotFoundError(f"Configured optimized router state does not exist: {path}")
    return router


def _make_extractors(
    proto: Dict[str, Any],
    guideline: Dict[str, Any],
) -> Dict[str, Any]:
    extractors = {}
    for section in ACTIVE_EXTRACTION_SECTIONS:
        allowed = SECTION_ALLOWED_ITEMS[section]
        fixed_spec = render_fixed_section_extraction_spec(section, allowed, proto, guideline)
        mutable_prompt = render_mutable_section_prototype_memory(section, allowed, proto)
        extractors[section] = GenericSectionExtractor(section, mutable_prompt, fixed_spec)
    return extractors


def _parse_extraction_records(raw: Any, section_name: str, article_input_json: str) -> List[Dict[str, Any]]:
    """Validate extractor output against allowed item IDs and canonical SIDs.

    As in v26, pnum returned by the model is not trusted; it is looked up from sid.
    """
    allowed = set(SECTION_ALLOWED_ITEMS.get(section_name, []))
    sid_lookup = sid_text_lookup_from_hierarchy(article_input_json)
    records: List[Dict[str, Any]] = []
    try:
        obj = extract_json_object(raw)
    except Exception:
        return records
    for ex in obj.get("extractions", []) or []:
        item = str(ex.get("item", "")).strip()
        sid = str(ex.get("sid", "")).strip()
        if item not in allowed or sid not in sid_lookup:
            continue
        records.append({
            "section_router": section_name,
            "item": item,
            "pnum": sid_lookup[sid]["pnum"],
            "sid": sid,
            "score": ex.get("score"),
        })
    return records


def _deduplicate_item_sid_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Merge raw section outputs into one item/sid relation set.

    Duplicates are merged while retaining all producer sections. No verifier/pruning is applied.
    """
    order = {s: i for i, s in enumerate(ACTIVE_EXTRACTION_SECTIONS)}
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    duplicates: List[Tuple[str, str]] = []

    for r in sorted(rows, key=lambda x: (order.get(x.get("section_router"), 999), _item_sort_key(x.get("item")), str(x.get("sid")))):
        key = (str(r["item"]), str(r["sid"]))
        if key in grouped:
            duplicates.append(key)
            sec = str(r.get("section_router", ""))
            if sec and sec not in grouped[key]["source_extractors"]:
                grouped[key]["source_extractors"].append(sec)
            continue
        base = dict(r)
        base["source_extractors"] = [str(r.get("section_router", ""))] if r.get("section_router") else []
        grouped[key] = base

    clean = list(grouped.values())
    clean.sort(key=lambda r: (_item_sort_key(r["item"]), int(r["pnum"]), str(r["sid"])))
    return clean, duplicates


def _build_demo_record(
    preprocessed: Dict[str, Any],
    pred_map: Dict[int, List[str]],
    router_diag: Dict[str, Any],
    article_context_json: str,
    anchor_diag: List[Dict[str, Any]],
    raw_rows: List[Dict[str, Any]],
    dedup_rows: List[Dict[str, Any]],
    duplicate_pairs: List[Tuple[str, str]],
    section_outputs: Dict[str, Dict[str, Any]],
    config: PipelineConfig,
) -> Dict[str, Any]:
    article_id = preprocessed["article_id"]
    pnum_to_pid = {int(p["pnum"]): str(p["pid"]) for p in preprocessed["paragraphs"]}
    sid_lookup = {
        str(s["sid"]): s
        for s in preprocessed["sentences"]
    }

    paragraphs = copy.deepcopy(preprocessed["paragraphs"])
    for p in paragraphs:
        p["predicted_sections"] = list(pred_map.get(int(p["pnum"]), []))

    final_extractions: List[Dict[str, Any]] = []
    for r in dedup_rows:
        sid = str(r["sid"])
        source = sid_lookup[sid]
        item = str(r["item"])
        final_extractions.append({
            "item": item,
            "official_section": ITEM_TO_SECTION.get(item, ""),
            "sid": sid,
            "pnum": int(source["pnum"]),
            "pid": str(source["pid"]),
            "text": str(source["text"]),
            "source_extractors": list(r.get("source_extractors", [])),
            "verification_status": "not_verified_v4",
            "score": r.get("score"),
        })

    counts: Dict[str, int] = {}
    for ex in final_extractions:
        counts[ex["item"]] = counts.get(ex["item"], 0) + 1

    active_items = {item for s in ACTIVE_EXTRACTION_SECTIONS for item in SECTION_ALLOWED_ITEMS[s]}
    checklist = []
    for item in ITEM_ORDER:
        active = item in active_items
        n = counts.get(item, 0)
        if not active:
            status = "not_evaluated"
        elif n:
            status = "reported"
        else:
            status = "potentially_missing"
        checklist.append({
            "item": item,
            "official_section": ITEM_TO_SECTION.get(item, ""),
            "status": status,
            "evidence_count": n,
            "active_in_pipeline": active,
            "excluded_from_test_metric": False,
        })

    reported = [r["item"] for r in checklist if r["status"] == "reported"]
    potentially_missing = [r["item"] for r in checklist if r["status"] == "potentially_missing"]
    not_evaluated = [r["item"] for r in checklist if r["status"] == "not_evaluated"]

    try:
        article_context = extract_json_object(article_context_json) if article_context_json else {k: [] for k in ARTICLE_CONTEXT_KEYS}
    except Exception:
        article_context = {k: [] for k in ARTICLE_CONTEXT_KEYS}

    section_predictions = [
        {"pnum": int(pnum), "pid": pnum_to_pid[int(pnum)], "sections": list(sections)}
        for pnum, sections in sorted(pred_map.items())
    ]

    serializable_section_outputs = {}
    for section, obj in section_outputs.items():
        serializable_section_outputs[section] = {
            "selected_pnums": obj.get("selected_pnums", []),
            "extraction_output_json": str(obj.get("extraction_output_json", "")),
            "validated_extractions": obj.get("validated_extractions", []),
        }

    id_warnings = []
    for s in preprocessed["sentences"]:
        if str(s["pid"]) != pnum_to_pid[int(s["pnum"])]:
            id_warnings.append({"kind": "pid_pnum_mismatch", "sid": s["sid"]})
    if len({s["sid"] for s in preprocessed["sentences"]}) != len(preprocessed["sentences"]):
        id_warnings.append({"kind": "duplicate_sid"})

    run_id = f"{article_id}__{preprocessed['content_hash']}__v4"
    return {
        "article_id": article_id,
        "run_id": run_id,
        "content_hash": preprocessed["content_hash"],
        "split": "live_upload",
        "pipeline_stage": "v26_live_preverifier_v4",
        "pipeline_config": {
            "model": config.model,
            "enable_article_context": bool(config.enable_article_context),
            "active_extraction_sections": list(ACTIVE_EXTRACTION_SECTIONS),
            "verifier_enabled": False,
            "prototype_path": str(config.prototype_path or ""),
            "annotation_guideline_path": str(config.annotation_guideline_path or ""),
            "optimized_router_path": str(config.optimized_router_path or ""),
        },
        "id_contract": preprocessed["id_contract"],
        "paragraphs": paragraphs,
        "sentences": copy.deepcopy(preprocessed["sentences"]),
        "section_predictions": section_predictions,
        "article_context": article_context,
        "final_extractions": final_extractions,
        "checklist": checklist,
        "summary": {
            "reported_items": reported,
            "potentially_missing_items": potentially_missing,
            "not_evaluated_items": not_evaluated,
            "n_final_extractions": len(final_extractions),
            "n_raw_section_extractions": len(raw_rows),
            "n_duplicate_raw_predictions": len(duplicate_pairs),
        },
        "raw_section_outputs": serializable_section_outputs,
        "verification": {
            "enabled": False,
            "reason": "Verifier/adjudication/recovery intentionally removed from v4 live pipeline.",
            "final_items": [
                {
                    "item": ex["item"],
                    "section": ex["official_section"],
                    "producer_sections": ex["source_extractors"],
                    "pnum": ex["pnum"],
                    "sid": ex["sid"],
                }
                for ex in final_extractions
            ],
        },
        "router_diagnostics": router_diag,
        "anchor_diagnostics": anchor_diag,
        "id_warnings": id_warnings,
    }


def check_asset_paths(config: PipelineConfig) -> Dict[str, Any]:
    """Describe whether v26 runtime assets are available."""
    prototype_ok = bool(config.prototype_path and Path(config.prototype_path).exists())
    guideline_ok = bool(config.annotation_guideline_path and Path(config.annotation_guideline_path).exists())
    router_requested = bool(config.optimized_router_path)
    router_ok = bool(config.optimized_router_path and Path(config.optimized_router_path).exists()) if router_requested else None
    return {
        "prototype": {"required": bool(config.strict_prototype), "path": config.prototype_path, "exists": prototype_ok},
        "annotation_guideline": {"required": False, "path": config.annotation_guideline_path, "exists": guideline_ok, "fallback_available": True},
        "optimized_router": {"required": False, "path": config.optimized_router_path, "exists": router_ok, "base_router_fallback": not router_requested},
    }


def run_txt_pipeline(
    text: str,
    article_id: str,
    api_key: str,
    config: Optional[PipelineConfig] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Run the v4 live path and return one demo-compatible article record."""
    config = config or PipelineConfig()
    progress = progress_callback or (lambda stage, detail: None)

    progress("preprocess", "Building deterministic paragraph/sentence hierarchy")
    preprocessed = preprocess_txt(text, article_id)
    article_input_json = preprocessed["article_input_json"]

    if config.strict_prototype and not (config.prototype_path and Path(config.prototype_path).exists()):
        raise FileNotFoundError(
            "The v26 CONSORT prototype JSON is required for faithful live extraction. "
            "Set PipelineConfig.prototype_path to consort_prototype_pass5_boundary_hardened_v1.json."
        )

    if not config.prototype_path:
        raise FileNotFoundError("No prototype_path was configured.")

    progress("assets", "Loading v26 prototype and annotation guideline")
    proto = load_consort_prototype(config.prototype_path)
    guideline = load_annotation_guideline(config.annotation_guideline_path)

    progress("model", f"Configuring DSPy model {config.model}")
    lm = _configure_dspy(api_key, config)

    with dspy.context(lm=lm):
        # Step 1: global paragraph router.
        progress("section_router", f"Routing {len(preprocessed['paragraphs'])} paragraphs")
        router = _make_router(config)
        section_pred = router(article_id=article_id, article_text=article_input_json)
        section_output_json = getattr(section_pred, "output_json", section_pred)
        pred_map, router_diag = validate_router_prediction(
            article_input_json,
            section_output_json,
            ACTIVE_EXTRACTION_SECTIONS,
        )

        # Article-level context memory, matching v26. Context never becomes evidence automatically.
        article_context_json = ""
        anchor_diag: List[Dict[str, Any]] = []
        if config.enable_article_context:
            progress("article_context", "Building grounded cross-section context memory")
            anchor = ArticleAnchorExtractor()
            anchor_pred = anchor(article_id=article_id, article_text=article_input_json)
            raw_context = getattr(anchor_pred, "output_json", anchor_pred)
            article_context_json, anchor_diag = sanitize_article_context_json(raw_context, article_input_json)

        # Step 2: independent section extractors. No verifier/recovery is instantiated.
        progress("extractors", "Running section-specific CONSORT extractors")
        extractors = _make_extractors(proto, guideline)
        section_outputs: Dict[str, Dict[str, Any]] = {}
        raw_rows: List[Dict[str, Any]] = []

        for section_name in ACTIVE_EXTRACTION_SECTIONS:
            selected_json = selected_paragraphs_for_target_section(article_input_json, pred_map, section_name)
            selected_pnums = sorted(parse_article_paragraphs_json(selected_json))
            if not selected_pnums:
                section_outputs[section_name] = {
                    "selected_pnums": [],
                    "selected_paragraphs_json": selected_json,
                    "extraction_output_json": json.dumps({
                        "article_id": article_id,
                        "section": section_name,
                        "extractions": [],
                    }),
                    "validated_extractions": [],
                }
                continue

            progress(section_name, f"Extracting from {len(selected_pnums)} routed paragraphs")
            pred = extractors[section_name](
                article_id=article_id,
                selected_paragraphs_json=selected_json,
                article_context_json=article_context_json,
                retry_feedback="",
            )
            raw_output = getattr(pred, "output_json", pred)
            validated = _parse_extraction_records(raw_output, section_name, article_input_json)
            raw_rows.extend(validated)
            section_outputs[section_name] = {
                "selected_pnums": selected_pnums,
                "selected_paragraphs_json": selected_json,
                "extraction_output_json": raw_output,
                "validated_extractions": validated,
            }

    progress("merge", "Validating exact SIDs and merging duplicate item/sentence pairs")
    dedup_rows, duplicate_pairs = _deduplicate_item_sid_rows(raw_rows)
    record = _build_demo_record(
        preprocessed=preprocessed,
        pred_map=pred_map,
        router_diag=router_diag,
        article_context_json=article_context_json,
        anchor_diag=anchor_diag,
        raw_rows=raw_rows,
        dedup_rows=dedup_rows,
        duplicate_pairs=duplicate_pairs,
        section_outputs=section_outputs,
        config=config,
    )
    progress("complete", f"Produced {len(record['final_extractions'])} item/sentence evidence pairs")
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess or run the CONSORT v4 live TXT pipeline.")
    parser.add_argument("txt", help="Input .txt article")
    parser.add_argument("--article-id", default=None)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--prototype", default=None)
    parser.add_argument("--guideline", default=None)
    parser.add_argument("--router", default=None)
    parser.add_argument("--model", default="openai/gpt-5.6-terra")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    p = Path(args.txt)
    article_id = args.article_id or p.stem
    text = p.read_text(encoding="utf-8", errors="replace")

    if args.preprocess_only:
        result = preprocess_txt(text, article_id)
        printable = {k: v for k, v in result.items() if k != "article_input_json"}
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        cfg = PipelineConfig(
            model=args.model,
            prototype_path=args.prototype,
            annotation_guideline_path=args.guideline,
            optimized_router_path=args.router,
        )
        result = run_txt_pipeline(text, article_id, api_key, cfg, lambda s, d: print(f"[{s}] {d}"))
        printable = result

    output = json.dumps(printable, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(output)
