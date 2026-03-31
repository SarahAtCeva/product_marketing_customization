from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from .config import PipelineConfig

_SEO_FIELDS = {
    "Titre produit", "Titre court du produit (60 caractères max)",
    "META Title", "META Description", "Description courte (...) 175 caractères",
    "Mots clés", "Argu 1", "Argu 2", "Argu 3", "Argu 4", "Argu 5",
}


def _response_text(response: Any) -> str:
    if text := getattr(response, "output_text", None):
        return text
    pieces = []
    for out_item in getattr(response, "output", []) or []:
        for content_item in getattr(out_item, "content", []) or []:
            if text := getattr(content_item, "text", None):
                pieces.append(text)
    return "\n".join(pieces) if pieces else str(response)


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
        if m:
            stripped = m.group(1).strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", stripped)
    if m:
        candidate = m.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON found in model response")


def _build_prompt(
    diff: dict[str, Any],
    channel_tone: dict[str, Any],
    channel: str,
    expertise: str,
    brief: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
) -> str:
    rows = diff.get("rows", [])
    payload = {
        "task": "Evaluate rewrite quality from old vs generated fields.",
        "requirements": [
            "Compute seo_improvement_ratio between 0 and 1.",
            "Grade channel_alignment_grade as one of A, B, C, D, E.",
            "Score grounding_alignment_score from 0 to 10 by checking generated values against the product brief and formulation_analysis — penalize any claim not supported there.",
            "Explain the channel grade in exactly 2 short lines.",
            "Provide exactly 6 actionable general remarks.",
            "Provide 1 one-line remark per generated field.",
            "Be strict and evidence-based using the provided diff rows.",
            "Flag any generated field that uses a forbidden claim from the compliance boundaries.",
        ],
        "scoring_guidance": {
            "seo_improvement_ratio": "0 = no improvement, 1 = major clear improvement.",
            "channel_alignment_grade": {"A": "Excellent fit", "B": "Good fit", "C": "Mixed", "D": "Weak", "E": "Poor / non-compliant"},
            "grounding_alignment_score": "10 = every claim traces to brief or formulation_analysis; 0 = hallucinated content.",
        },
        "channel": channel,
        "expertise": expertise,
        "channel_profile": (channel_tone.get("channels") or {}).get(channel),
        "expertise_profile": (channel_tone.get("expertise_profiles") or {}).get(expertise),
        "product_brief": brief,
        "compliance_boundaries": compliance,
        "diff_summary": diff.get("summary"),
        "seo_rows": [r for r in rows if r.get("field") in _SEO_FIELDS],
        "all_rows": rows,
        "expected_per_field_remarks_keys": [r.get("field") for r in rows if r.get("field")],
        "output_constraints": [
            "Return JSON only",
            "channel_alignment_why_2_lines must be exactly length 2",
            "general_remarks must be exactly length 6",
            "per_field_remarks must include every key in expected_per_field_remarks_keys",
            "Each per-field remark must be a single short line",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_schema(expected_fields: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "seo_improvement_ratio", "channel_alignment_grade", "grounding_alignment_score",
            "channel_alignment_why_2_lines", "general_remarks", "per_field_remarks", "score_breakdown",
        ],
        "properties": {
            "seo_improvement_ratio": {"type": "number", "minimum": 0, "maximum": 1},
            "channel_alignment_grade": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
            "grounding_alignment_score": {"type": "number", "minimum": 0, "maximum": 10},
            "channel_alignment_why_2_lines": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
            "general_remarks": {"type": "array", "items": {"type": "string"}, "minItems": 6, "maxItems": 6},
            "per_field_remarks": {
                "type": "object",
                "additionalProperties": False,
                "properties": {f: {"type": "string"} for f in expected_fields},
                "required": list(dict.fromkeys(expected_fields)),
            },
            "score_breakdown": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title_and_meta", "keyword_usage", "clarity_and_readability", "compliance_safety", "channel_fit"],
                "properties": {k: {"type": "number", "minimum": 0, "maximum": 1}
                               for k in ["title_and_meta", "keyword_usage", "clarity_and_readability", "compliance_safety", "channel_fit"]},
            },
        },
    }


def judge(
    diff: dict[str, Any],
    cfg: PipelineConfig,
    brief: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with cfg.channel_tone_path.open("r", encoding="utf-8") as f:
        channel_tone = json.load(f)

    expected_fields = [r.get("field") for r in diff.get("rows", []) if r.get("field")]
    prompt = _build_prompt(diff, channel_tone, cfg.channel, cfg.expertise, brief, compliance)
    schema = _build_schema(expected_fields)
    cfg.save_prompt_debug("judge", {"model": cfg.reasoning_model, "max_output_tokens": 4000, "messages": [{"role": "user", "content": prompt}]})

    client = OpenAI()
    response = client.responses.create(
        model=cfg.reasoning_model,
        input=prompt,
        max_output_tokens=4000,
        text={"format": {"type": "json_schema", "name": "judge_output", "schema": schema, "strict": True}},
    )

    raw = _response_text(response)
    return json.loads(_extract_json(raw))
