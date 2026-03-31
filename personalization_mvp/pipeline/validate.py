from __future__ import annotations

import re
from typing import Any

_BANNED_CLAIM_PATTERNS = [
    r"\bgu[eé]rit?\b",
    r"\bcure\b",
    r"\b100\s?%\b",
    r"\bgaranti(?:e|es|s)?\b",
    r"\bimm[eé]diat(?:e|ement)?\b",
    r"\bmiracle\b",
    r"\bsans\s+risque\b",
]
_VOLUME_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\s?(?:ml|cl|l|g|kg)\b", re.IGNORECASE)


def sanitize(value: str) -> str:
    for old, new in {"\u0092": "'", "\u0095": "- ", "\u0085": "...", "\u0096": "-", "\u00a0": " ", "\r": "\n"}.items():
        value = value.replace(old, new)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def coerce_keywords(value: Any) -> list[str]:
    if isinstance(value, list):
        parts = [str(v) for v in value if str(v).strip()]
    elif isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    else:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        norm = re.sub(r"\s+", " ", part).strip(" ,;.")
        key = norm.lower()
        if norm and key not in seen:
            seen.add(key)
            out.append(norm)
    return out


def validate_seo_fields(
    generated_fields: dict[str, str],
    specs: dict[str, Any],
    product: dict[str, Any],
) -> dict[str, Any]:
    fields_spec = specs.get("fields") or {}
    normalized: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    core = product.get("core_identity") or {}
    brand = (core.get("Marque") or product.get("brand") or "").strip()

    for field, spec in fields_spec.items():
        raw = generated_fields.get(field, "")
        value = sanitize(str(raw)) if raw is not None else ""

        if field == "Mots clés":
            keywords = coerce_keywords(value)
            value = ", ".join(keywords)
            if len(keywords) < 5:
                warnings.append({"field": field, "rule": "keyword_count", "message": "Low keyword coverage."})
            for kw in keywords:
                if len(kw.split()) > 5:
                    warnings.append({"field": field, "rule": "keyword_length", "message": f"Keyword '{kw}' is too long."})

        if not value:
            errors.append({"field": field, "rule": "required", "message": "Field is empty."})
            normalized[field] = value
            continue

        if isinstance(spec, dict):
            min_c = spec.get("min_chars")
            max_c = spec.get("max_chars")
            if isinstance(min_c, int) and len(value) < min_c:
                errors.append({"field": field, "rule": "min_chars", "message": f"Expected >= {min_c} chars, got {len(value)}."})
            if isinstance(max_c, int) and len(value) > max_c:
                errors.append({"field": field, "rule": "max_chars", "message": f"Expected <= {max_c} chars, got {len(value)}."})

        if field == "Titre produit" and brand and not value.lower().startswith(brand.lower()):
            errors.append({"field": field, "rule": "brand_first", "message": f"Title must start with brand '{brand}'."})

        if field == "Titre court du produit (60 caractères max)" and _VOLUME_RE.search(value):
            errors.append({"field": field, "rule": "no_volume", "message": "Short title must not include volume/weight."})

        if field == "META Description":
            if len([s for s in re.split(r"[.!?]+", value) if s.strip()]) < 2:
                errors.append({"field": field, "rule": "two_sentences", "message": "META Description needs at least two sentences."})

        for pattern in _BANNED_CLAIM_PATTERNS:
            if re.search(pattern, value, flags=re.IGNORECASE):
                errors.append({"field": field, "rule": "compliance_claim", "message": f"Non-compliant claim: '{pattern}'."})
                break

        normalized[field] = value

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "normalized_fields": normalized,
    }


def summarize_validation_issues(validation: dict[str, Any], limit: int = 12) -> str:
    errors = validation.get("errors") or []
    warnings = validation.get("warnings") or []
    lines: list[str] = []
    for issue in errors[:limit]:
        lines.append(f"ERROR [{issue.get('field')}::{issue.get('rule')}]: {issue.get('message')}")
    for issue in warnings[: max(0, limit - len(lines))]:
        lines.append(f"WARN [{issue.get('field')}::{issue.get('rule')}]: {issue.get('message')}")
    return "\n".join(lines) if lines else "No validation issues."


def compute_diff(
    product: dict[str, Any],
    generated_fields: dict[str, Any],
    field_specs: dict[str, Any],
) -> dict[str, Any]:
    """Side-by-side comparison of original CSV values vs generated values."""
    old: dict[str, Any] = {}
    for section in ("commercial_seo_inputs", "content_body_inputs"):
        block = product.get(section) or {}
        if isinstance(block, dict):
            old.update(block)

    _aliases: dict[str, list[str]] = {
        "Titre court du produit (60 caractères max)": ["Titre court", "Titre court du produit (60 caractères max)"],
        "Description courte (...) 175 caractères": ["Description courte (175 caractères)"],
        "Description du produit unique/site 400 caractères": ["Description du produit unique/site 400 caractères"],
    }

    def resolve(field: str) -> Any:
        if field in old:
            return old[field]
        for alias in _aliases.get(field, []):
            if alias in old:
                return old[alias]
        return None

    target = list((field_specs.get("fields") or {}).keys()) or list(generated_fields.keys())
    rows: list[dict[str, Any]] = []
    for field in target:
        old_val = resolve(field)
        new_val = generated_fields.get(field)
        rows.append({
            "field": field,
            "old_value": old_val,
            "generated_value": new_val,
            "changed": (str(old_val or "").strip() != str(new_val or "").strip()),
        })

    changed = sum(1 for r in rows if r["changed"])
    return {
        "summary": {"total_fields": len(rows), "changed_fields": changed, "unchanged_fields": len(rows) - changed},
        "rows": rows,
    }
