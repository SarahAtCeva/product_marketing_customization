from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from .config import PipelineConfig


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
    judgment: dict[str, Any],
    field_specs: dict[str, Any],
    channel_tone: dict[str, Any],
    channel: str,
    expertise: str,
    brief: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    # Only retry changed fields
    rows = [r for r in diff.get("rows", []) if r.get("changed") is True]
    target_fields = [r["field"] for r in rows if r.get("field")]

    per_field_remarks = judgment.get("per_field_remarks") or {}
    all_specs = field_specs.get("fields") or {}

    prompt_obj = {
        "task": "Revise generated ecommerce copy fields using judge feedback.",
        "rules": [
            "Apply all general feedback remarks.",
            "If the original title/short title / Meta data has a strong keyword about functionality , try to keep it not rephrased as it was intentional "
            "Apply each field-specific remark to its field.",
            "Respect field-level constraints strictly.",
            "Respect channel and expertise rules strictly.",
            "Every claim must trace to the product brief or formulation_analysis — do not introduce new facts.",
            "Only use allowed_claims from compliance_boundaries. Never use forbidden_claims.",
            "Do not invent scientific/medical facts.",
            "Return JSON only.",
        ],
        "channel": channel,
        "expertise": expertise,
        "channel_profile": (channel_tone.get("channels") or {}).get(channel),
        "expertise_profile": (channel_tone.get("expertise_profiles") or {}).get(expertise),
        "product_brief": brief,
        "compliance_boundaries": compliance,
        "field_specifications": {k: v for k, v in all_specs.items() if k in target_fields},
        "judge_feedback": {
            "seo_improvement_ratio": judgment.get("seo_improvement_ratio"),
            "channel_alignment_grade": judgment.get("channel_alignment_grade"),
            "grounding_alignment_score": judgment.get("grounding_alignment_score"),
            "channel_alignment_why_2_lines": judgment.get("channel_alignment_why_2_lines"),
            "general_remarks": judgment.get("general_remarks", []),
            "per_field_remarks": {f: per_field_remarks.get(f) for f in target_fields},
        },
        "inputs": {
            "old_fields": {r["field"]: r.get("old_value") for r in rows if r.get("field")},
            "current_generated_fields": {r["field"]: r.get("generated_value") for r in rows if r.get("field")},
            "target_fields": target_fields,
        },
        "output_schema": {"generated_fields": {f: "<revised string>" for f in target_fields}},
    }
    return json.dumps(prompt_obj, ensure_ascii=False, indent=2), target_fields


def retry_fields(
    diff: dict[str, Any],
    judgment: dict[str, Any],
    cfg: PipelineConfig,
    brief: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with cfg.field_specs_path.open("r", encoding="utf-8") as f:
        field_specs = json.load(f)
    with cfg.channel_tone_path.open("r", encoding="utf-8") as f:
        channel_tone = json.load(f)

    prompt, target_fields = _build_prompt(
        diff=diff,
        judgment=judgment,
        field_specs=field_specs,
        channel_tone=channel_tone,
        channel=cfg.channel,
        expertise=cfg.expertise,
        brief=brief,
        compliance=compliance,
    )
    cfg.save_prompt_debug("retry", {"model": cfg.generation_model, "max_output_tokens": 2200, "messages": [{"role": "user", "content": prompt}]})

    client = OpenAI()
    response = client.responses.create(
        model=cfg.generation_model,
        input=prompt,
        max_output_tokens=2200,
    )

    raw = _response_text(response)
    data = json.loads(_extract_json(raw))

    gen = data.get("generated_fields")
    if not isinstance(gen, dict):
        raise ValueError("retry output must contain a 'generated_fields' object")

    missing = [f for f in target_fields if f not in gen]
    if missing:
        raise ValueError(f"retry output missing fields: {missing}")

    return data
