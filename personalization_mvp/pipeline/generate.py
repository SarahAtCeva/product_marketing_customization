from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import PipelineConfig
from .validate import coerce_keywords, sanitize, summarize_validation_issues, validate_seo_fields


# ── shared helpers ────────────────────────────────────────────────────────────

def _slim_product_for_descriptions(product: dict[str, Any]) -> dict[str, Any]:
    """Return only identity fields and existing copy values — brief/compliance cover the rest."""
    core = product.get("core_identity") or {}
    commercial = product.get("commercial_seo_inputs") or {}
    content = product.get("content_body_inputs") or {}
    identity = {k: core[k] for k in ("Marque", "Gamme", "Espèce", "Présentation", "Designation") if core.get(k)}
    existing = {k: v for k, v in {
        **{k: v for k, v in content.items() if v},
        **{f"Argu {i}": commercial[f"Argu {i}"] for i in range(1, 6) if commercial.get(f"Argu {i}")},
    }.items()}
    return {"identity": identity, "existing_fields": existing}

def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_json(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def _channel_profile_text(
    channel_tone: dict[str, Any], channel: str, expertise: str
) -> str:
    channel_cfg = (channel_tone.get("channels") or {}).get(channel, {})
    expertise_cfg = (channel_tone.get("expertise_profiles") or {}).get(expertise, {})
    style_rules = channel_cfg.get("style_rules", {})
    tone_voice = style_rules.get("tone") or expertise_cfg.get("tone", {}).get("voice") or "professionnel et rassurant"
    motivations = style_rules.get("buyer_motivations", [])
    motivation_text = ", ".join(str(x) for x in motivations if str(x).strip()) or "sécurité, efficacité, rapport qualité-prix"
    return (
        f"Canal e-commerce: {channel}\n"
        f"Profil expertise: {expertise}\n"
        f"Ton attendu: {tone_voice}\n"
        f"Motivations principales: {motivation_text}\n"
        "Prioriser précision factuelle, conformité, lisibilité et bénéfices concrets."
    )


# ── descriptions generation ───────────────────────────────────────────────────

def _load_examples(examples_dir: Path) -> dict[str, list[str]]:
    if not examples_dir.exists():
        return {}
    grouped: dict[str, list[str]] = {}
    for fp in sorted(examples_dir.glob("*.txt")):
        content = fp.read_text(encoding="utf-8").strip()
        if not content:
            continue
        low = fp.name.lower()
        if "description_courte" in low:
            field = "Description courte 600 caractères"
        elif "description_longue" in low:
            field = "Description longue"
        else:
            field = "Autres modèles"
        grouped.setdefault(field, []).append(f"[Source: {fp.name}]\n{content}")
    return grouped


def _fmt_examples(examples: dict[str, list[str]]) -> str:
    if not examples:
        return "Aucun modèle d'exemple fourni."
    chunks: list[str] = []
    for field_name, samples in examples.items():
        chunks.append(f"Champ: {field_name}")
        chunks.extend(samples)
    return "\n\n".join(chunks)


def generate_descriptions(
    product: dict[str, Any],
    cfg: PipelineConfig,
    brief: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    channel_tone = _load_json(cfg.channel_tone_path)
    channel_profile = _channel_profile_text(channel_tone, cfg.channel, cfg.expertise)
    specs = _load_json(cfg.descriptions_specs_path)
    examples = _load_examples(cfg.field_specs_path.parent / "descriptions_content" / "examples")

    system = (
        "You are a senior French ecommerce veterinary copywriter. "
        "Make use of the content in the provided product Info to be convincing and synthetic"
        "The writing should be natural and fluid ,the purpose is to make excellent content that is compliant to the selling channel"
        "Before generating any field , remember its length constraints and specifications."
        "Respect strictly the length constraints ,plan ahead to do so."
        "The channel profile is provided and it gives you an idea about the people the content is intended for.\n"
        "Rules:\n"
        "1) Never invent facts, efficacy rates, certifications, durations, ingredients, or warnings.\n"
        "2) If you don't respect the field specifications , all that you generate is useless.\n "
        "3) You should repect the length constraint for each field to generate.\n"
        "4) Every sentence must trace back to a fact in the product brief or product data.\n"
        "5) Strictly respect the compliance boundaries: use only allowed claims, never forbidden ones.\n"
        "6) Respect Strictly every field specification (char limits, structure, list size), otherwise the content is not accepted\n"
        "7) Reproduce mandatory mentions faithfully when relevant.\n"
        "8) Tone must be professional, clear, and reassuring.\n"
        "9) Prioritize buyer motivations from channel context.\n"
        "10) Remove redundancy and fix broken encoding artifacts.\n"
       
        "11) Output strict JSON only, no markdown, no commentary.\n"
        "12) If required info is missing, keep text factual; do not hallucinate.\n"
        "13) If you don't respect the field specifications , all that you generate is useless.\n "
        "14) For the field  'Description longue', the writing should be natural , structured and convincing, do not exceed 2000 caracters"
        "It s a very important field as it is what is going to be displayed in the website.\n "
  
    )
    user = (
        "Produce JSON with exactly these top-level keys:\n"
        "- Description du produit unique/site 400 caractères\n"
        "- Description courte 600 caractères\n"
        "- Description longue\n"
        "- Arguments de vente uniques (USPs) 3 à 5 \n"
        "- self_check\n\n"
        "For 'Arguments de vente uniques (USPs) 3 à 5', return an array with 3 to 5 strings , make sure they rephrase every existing argument if it exists."
        "If no existing argurments are available then generate them factually.\n"

        
        "For 'self_check', return an object with: char_counts, spec_compliance, "
        "regulatory_consistency, channel_fit, hallucination_risk.\n\n"
        "Examples policy: Use examples as structural/tone templates only. "
        "Never copy product-specific facts from examples.\n\n"
        f"Input: channel profile\n{channel_profile}\n\n"
        + (f"Input: product brief\n{json.dumps(brief, ensure_ascii=False, indent=2)}\n\n" if brief else "")
        + (f"Input: compliance boundaries\n{json.dumps(compliance, ensure_ascii=False, indent=2)}\n\n" if compliance else "")
        + f"Input: field specifications\n{json.dumps(specs, ensure_ascii=False, indent=2)}\n\n"
        f"Input: Some good long description examples to use as inspo\n{_fmt_examples(examples)}\n\n"
       
    )

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    cfg.save_prompt_debug("generate_descriptions", {"model": cfg.generation_model, "temperature": 0.2, "messages": messages})

    client = OpenAI()
    response = client.responses.create(
        model=cfg.generation_model,
        input=messages,
        temperature=0.2,
        text={"format": {"type": "json_object"}},
    )
    return json.loads(response.output_text)


# ── SEO generation (single candidate + one repair if validation fails) ────────

def _normalize_product_for_seo(raw: dict[str, Any]) -> dict[str, Any]:
    core = raw.get("core_identity") or {}
    commercial = raw.get("commercial_seo_inputs") or {}
    content = raw.get("content_body_inputs") or {}

    def first(*vals: Any) -> str:
        for v in vals:
            if isinstance(v, str) and v.strip():
                return sanitize(v)
        return ""

    title = first(
        commercial.get("Titre produit"),
        commercial.get("Titre court"),
        commercial.get("META Title"),
        core.get("Designation"),
    )
    summary = first(
        content.get("Description courte (175 caractères)"),
        content.get("Description du produit unique/site 400 caractères"),
        content.get("Description courte 600 caractères"),
        commercial.get("META Description"),
    )
    if not title or not summary:
        return {}

    norm = dict(raw)
    norm["title"] = title
    norm["summary"] = summary
    category = first(core.get("Catégorie (Clients)"), core.get("Catégorisation"))
    brand = first(core.get("Marque"))
    if category:
        norm["category"] = category
    if brand:
        norm["brand"] = brand
    return norm


def _seo_target_fields(specs: dict[str, Any]) -> list[str]:
    return list((specs.get("fields") or {}).keys())


def _normalize_seo_output(data: dict[str, Any], target_fields: list[str]) -> dict[str, str]:
    raw = data.get("generated_fields") if isinstance(data.get("generated_fields"), dict) else data
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for field in target_fields:
        value = raw.get(field, "")
        if field == "Mots clés":
            out[field] = ", ".join(coerce_keywords(value))
        else:
            out[field] = sanitize(str(value)) if value is not None else ""
    return out


def _seo_generate_prompt(
    product: dict[str, Any],
    specs: dict[str, Any],
    brief: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
) -> str:
    core = product.get("core_identity") or {}
    commercial = product.get("commercial_seo_inputs") or {}
    scientific = product.get("scientific_regulatory_inputs") or {}
    content = product.get("content_body_inputs") or {}
    context = {
        "brand": (core.get("Marque") or product.get("brand") or "").strip(),
        "range": (core.get("Gamme") or "").strip(),
        "species": (core.get("Espèce") or "").strip(),
        "format": (core.get("Présentation") or "").strip(),
        "category": (core.get("Catégorie (Clients)") or product.get("category") or "").strip(),
        "title": product.get("title", ""),
        "summary": product.get("summary", ""),
        "warnings": (scientific.get("Contre-indication, avertissements, mentions obligatoires") or "").strip(),
        "usage": (
            content.get("Conseils d'utilisation uniques (SEO)")
            or scientific.get("Conseils d'utilisation institutionnels")
            or ""
        ).strip(),
        "existing_keywords": coerce_keywords(commercial.get("Mots clés") or ""),
    }
    target_fields = _seo_target_fields(specs)
    fields_schema = ", ".join(
        f'{json.dumps(f, ensure_ascii=False)}: "<string>"' for f in target_fields
    )

    return (
        "You are a senior French SEO ecommerce copywriter specialized in animal health products.\n\n"
        "Generate ONE complete set of SEO fields in French for this product. Remember their constraints before starting to generate.All requested fields are mandatory.\n\n"
        "Primary goal:\n"
        "- Maximize real ecommerce conversion while staying strictly compliant and not exceeding caracter count limits per field.\n\n"
        "Priority order (mandatory):\n"
        "1. Compliance and safety\n"
        "2. Respect of field specifications and formatting\n"
        "3. Fidelity to the product data\n"
        "4. Conversion and SEO performance\n"
        "5. Natural, fluent French\n\n"
        "Hard requirements:\n"
        "- Respect the field specifications exactly. Otherwise the output is unusable.\n"
        "- Strictly follow the compliance boundaries: use only allowed claims, never forbidden ones.\n"
        "- Do not invent facts, ingredients, effects, promises, species, formats, or usage contexts not supported by the input.\n"
        "- Keep language natural and persuasive, never robotic, never keyword stuffing.\n"
        "- If the title or metadata contain a strong central keyword about product function or type, preserve it naturally in the generated fields.\n"
        "- If the title has a good inclusion of product content and some good product specifities,numbers also preserve it improve on it."
        "- Product title must start with brand name when brand is available.\n"
        "- For short title: do not mention volume/weight (ml, cl, l, g, kg, etc.).\n"
        "- If some data is missing, write the safest accurate version using only available information.\n\n"
        "Conversion writing rules:\n"
        "- Write to help the customer choose the product quickly and confidently.\n"
        "- Highlight the main user need addressed by the product when supported by the input.\n"
        "- Prefer compliant benefit language such as 'aide à', 'contribue à', 'participe à', 'convient à', 'idéal pour' when relevant.\n"
        "- Reduce hesitation by making the product feel clear, relevant, and easy to understand.\n"
        "- Prefer concrete, specific wording over vague marketing language.\n"
        "- Avoid empty claims like 'premium', 'exceptionnel', 'révolutionnaire' unless explicitly supported.\n"
        "- Avoid medical, absolute, guaranteed, miraculous, or immediate-effect language.\n\n"
        "Working method:\n"
        "1. Identify the product's core identity: brand, range, species, format, category.\n"
        "2. Identify the main keyword from the original title, category, summary, and existing keywords.\n"
        "3. Identify the most relevant purchase intent and express it in a compliant way.\n"
        "4. Generate each field according to its own constraints.\n"
        "5. Check consistency across all fields.\n"
        "6. Check again that every field is compliant, natural, and correctly formatted.\n\n"
        f"Product context:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        + (f"Additional brief:\n{json.dumps(brief, ensure_ascii=False, indent=2)}\n\n" if brief else "")
        + (f"Compliance boundaries:\n{json.dumps(compliance, ensure_ascii=False, indent=2)}\n\n" if compliance else "")
        + f"Field specs:\n{json.dumps(specs.get('fields', {}), ensure_ascii=False, indent=2)}\n\n"
        f'Return STRICT JSON only:\n{{"generated_fields": {{{fields_schema}}}}}\n'
    )
def _seo_repair_prompt(
    specs: dict[str, Any], fields: dict[str, str], feedback: str
) -> str:
    target_fields = _seo_target_fields(specs)
    fields_schema = ", ".join(f'{json.dumps(f, ensure_ascii=False)}: "<string>"' for f in target_fields)
    return (
        "You are fixing French ecommerce SEO fields.\n\n"
        f"Validation issues to fix:\n{feedback}\n\n"
        f"Current fields:\n{json.dumps(fields, ensure_ascii=False, indent=2)}\n\n"
        f"Field specs:\n{json.dumps(specs.get('fields', {}), ensure_ascii=False, indent=2)}\n\n"
        f'Return STRICT JSON only:\n{{"generated_fields": {{{fields_schema}}}}}\n'
    )


def generate_seo(
    product: dict[str, Any],
    cfg: PipelineConfig,
    brief: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    specs = _load_json(cfg.seo_specs_path)
    target_fields = _seo_target_fields(specs)
    norm_product = _normalize_product_for_seo(product)
    if not norm_product:
        raise ValueError("Cannot normalize product for SEO: missing title or summary.")

    client = OpenAI()

    seo_prompt = _seo_generate_prompt(norm_product, specs, brief, compliance)
    cfg.save_prompt_debug("generate_seo", {"model": cfg.reasoning_model, "temperature": 0.2, "messages": [{"role": "user", "content": seo_prompt}]})
    response = client.chat.completions.create(
        model=cfg.reasoning_model,
        messages=[{"role": "user", "content": seo_prompt}],
        temperature=0.2,
    )
    fields = _normalize_seo_output(_safe_json(response.choices[0].message.content) or {}, target_fields)

    validation = validate_seo_fields(fields, specs, norm_product)
    if not validation["is_valid"]:
        feedback = summarize_validation_issues(validation)
        repair_prompt = _seo_repair_prompt(specs, fields, feedback)
        cfg.save_prompt_debug("generate_seo_repair", {"model": cfg.reasoning_model, "temperature": 0.4, "messages": [{"role": "user", "content": repair_prompt}]})
        repair_resp = client.chat.completions.create(
            model=cfg.reasoning_model,
            messages=[{"role": "user", "content": repair_prompt}],  # compliance already baked into fields context
            temperature=0.4,
        )
        repaired = _normalize_seo_output(_safe_json(repair_resp.choices[0].message.content) or {}, target_fields)
        if repaired:
            fields = repaired
            validation = validate_seo_fields(fields, specs, norm_product)

    return {
        "generated_fields": fields,
        "validation": {
            "is_valid": validation["is_valid"],
            "errors": validation.get("errors", []),
            "warnings": validation.get("warnings", []),
        },
    }


# ── merge + combined entry point ──────────────────────────────────────────────

_DESCRIPTION_FIELDS = [
    "Description du produit unique/site 400 caractères",
    "Description courte 600 caractères",
    "Description longue",
]


def generate(
    product: dict[str, Any],
    cfg: PipelineConfig,
    brief: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate descriptions + SEO and merge into a single generated_fields dict."""
    descriptions = generate_descriptions(product, cfg, brief, compliance)
    seo = generate_seo(product, cfg, brief, compliance)

    merged: dict[str, Any] = dict(seo.get("generated_fields") or {})

    for field in _DESCRIPTION_FIELDS:
        value = descriptions.get(field)
        if value and str(value).strip():
            merged[field] = value

    usps = descriptions.get("Arguments de vente uniques (USPs) 3 à 5")
    if isinstance(usps, list):
        for idx in range(5):
            key = f"Argu {idx + 1}"
            if idx < len(usps) and str(usps[idx]).strip():
                merged[key] = usps[idx]

    # Fill any remaining target fields from original CSV values
    field_specs = _load_json(cfg.field_specs_path)
    existing: dict[str, Any] = {}
    for section in ("commercial_seo_inputs", "content_body_inputs"):
        block = product.get(section) or {}
        if isinstance(block, dict):
            existing.update(block)

    _aliases = {
        "Titre court du produit (60 caractères max)": ["Titre court"],
        "Description courte (...) 175 caractères": ["Description courte (175 caractères)"],
    }

    def resolve_existing(field: str) -> Any:
        if field in existing:
            return existing[field]
        for alias in _aliases.get(field, []):
            if alias in existing:
                return existing[alias]
        return ""

    for field in (field_specs.get("fields") or {}).keys():
        if str(merged.get(field, "")).strip():
            continue
        merged[field] = resolve_existing(field)

    return {
        "generated_fields": merged,
        "descriptions_self_check": descriptions.get("self_check"),
        "seo_validation": seo.get("validation"),
    }
