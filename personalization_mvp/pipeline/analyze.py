from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import PipelineConfig


_SYSTEM = (
    "You are a senior veterinary product analyst, animal health copywriting strategist, "
    "and French regulatory compliance expert.\n\n"

    "YOUR PURPOSE: You produce the single analytical document that a downstream LLM will "
    "use as its ONLY source of truth to generate a compelling, compliant product description "
    "for a French pet health e-commerce site.\n\n"

    "WHAT THIS MEANS IN PRACTICE:\n"
    "- Every field you fill becomes raw material for persuasive copy. "
    "If you leave something vague, the generator will either invent (dangerous) or omit (lost sale).\n"
    "- The generator has NO access to the original product data — only your output. "
    "So you must extract, interpret, and pre-digest everything it needs.\n"
    "- You are the last line of defense against unsupported claims. "
    "If you mark something as allowed, the generator WILL use it.\n\n"

    "YOUR ANALYTICAL STANCE:\n"
    "- Be exhaustive in extraction: scan every single field, including obscure ones. "
    "Product data is messy — critical selling points often hide in 'Conseils d'utilisation' "
    "or 'En savoir plus' rather than in the main description.\n"
    "- Be precise in language: prefer concrete, benefit-oriented phrasing over generic labels. "
    "'Soulage les démangeaisons liées aux allergies saisonnières' beats 'action apaisante'.\n"
    "- Be honest about gaps: a clearly flagged data gap is more valuable than a padded answer. "
    "The generator can work around a gap; it cannot recover from a hallucinated claim.\n"
    "- Think like the pet owner: every buyer_question, use_case, and benefit should reflect "
    "real purchase-decision moments — the 2 AM Google search, the vet visit follow-up, "
    "the comparison between two products in a cart.\n\n"

    "STRICT RULES:\n"
    "- Ground every claim in the provided data. Never invent benefits, ingredients, "
    "certifications, dosages, or efficacy outcomes.\n"
    "- Distinguish clearly between manufacturer marketing claims and verifiable facts.\n"
    "- Distinguish clearly between manufacturer claims and actual user/customer feedback.\n"
    "- Apply French and EU regulatory frameworks for the product's category "
    "(médicament vétérinaire, complément alimentaire, biocide, aliment diététique, "
    "dispositif de soin, etc.).\n"
    "- When in doubt about a claim's permissibility, classify it as forbidden."
)


def build_prompt(product_json: str, specs: dict[str, Any]) -> str:
    steps = specs.get("steps", {})
    steps_text = ""
    for key in ("etape_1", "etape_2", "etape_3", "etape_4"):
        step = steps.get(key, {})
        objectif = step.get("objectif", "")
        rules = step.get("critical_rules", [])
        label = key.replace("etape_", "ÉTAPE ").upper()
        steps_text += f"━━━ {label} ━━━\n"
        steps_text += f"Objectif : {objectif}\n"
        for rule in rules:
            steps_text += f"• {rule}\n"
        steps_text += "\n"

    return (
        "Analyse le produit de santé animale ci-dessous. "
        "Suis les 4 étapes dans l'ordre strict — chaque étape alimente la suivante.\n\n"
        f"{steps_text}"
        "━━━ DONNÉES PRODUIT ━━━\n"
        f"{product_json}\n"
    )


_BRIEF_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "product_summary",
        "target_species",
        "target_profile",
        "primary_benefit",
        "secondary_benefits",
        "use_cases",
        "key_differentiator",
        "tone_angle",
        "data_gaps",
        "buyer_questions",
        "copy_angles",
        "formulation_analysis",
        "existing_arguments",
        "explanation_of_the_concept_innovation",
        "efficacy_claims",
        "scientific_evidence",
        "user_feedback",
    ],
    "properties": {
        "product_summary": {
            "type": "string",
            "description": "1-2 sentence factual description of what this product is and does.",
        },
        "target_species": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Animals this product is intended for (e.g. chat, chien, lapin).",
        },
        "target_profile": {
            "type": "string",
            "description": "Specific animal profile: age, size, condition, lifestyle if known.",
        },
        "primary_benefit": {
            "type": "string",
            "description": "The single most important benefit this product delivers.",
        },
        "secondary_benefits": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Additional benefits, each as a short phrase.",
        },
        "use_cases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete situations where a pet owner would use this product.",
        },
        "key_differentiator": {
            "type": "string",
            "description": "What makes this product stand out vs. the general category.",
        },
        "tone_angle": {
            "type": "string",
            "enum": ["clinical", "reassuring", "educational", "conversion"],
            "description": "Recommended communication tone based on product nature.",
        },
        "data_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Fields that are empty or too vague to support strong copy.",
        },
        "buyer_questions": {
            "type": "array",
            "description": "Prioritized questions a buyer would ask, flagged as answered or not.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["priority", "question", "answered", "answer_summary"],
                "properties": {
                    "priority": {
                        "type": "integer",
                        "description": "1 = most important buyer concern.",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question as a buyer would phrase it, in French.",
                    },
                    "answered": {
                        "type": "boolean",
                        "description": "True if existing product data answers this question.",
                    },
                    "answer_summary": {
                        "type": "string",
                        "description": (
                            "Short answer from product data, or "
                            "'Information absente des données produit.' if not answered."
                        ),
                    },
                },
            },
        },
        "existing_arguments": {
            "type": "array",
            "description": "Verbatim content of Argu 1–5 fields found in the product data, in order. Omit empty fields.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "content"],
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "Field name, e.g. 'Argu 1'.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Verbatim content of the field.",
                    },
                },
            },
        },
        "explanation_of_the_concept_innovation": {
            "type": "string",
            "description": "The medical idea or innovation about the product if mentioned.",
        },
        "formulation_analysis": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "parsed_active_ingredients",
                "inferred_functional_properties",
                "text_field_mentions",
                "unsupported_claims_risk",
            ],
            "description": (
                "Explicit reasoning over formulation data (structured fields + free-text paragraphs). "
                "Must be completed before deriving primary_benefit or secondary_benefits."
            ),
            "properties": {
                "parsed_active_ingredients": {
                    "type": "array",
                    "description": "Ingredients/additives identified in Composition or Additifs nutritionnels.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "source_field", "quantity_or_context"],
                        "properties": {
                            "name": {"type": "string"},
                            "source_field": {
                                "type": "string",
                                "enum": ["Composition", "Additifs nutritionnels", "other"],
                            },
                            "quantity_or_context": {
                                "type": "string",
                                "description": "Dosage, unit, or 'non précisé' if absent.",
                            },
                        },
                    },
                },
                "inferred_functional_properties": {
                    "type": "array",
                    "description": (
                        "Properties legitimately inferred from identified ingredients. "
                        "Each must cite the ingredient that supports it."
                    ),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["property", "supporting_ingredient", "confidence"],
                        "properties": {
                            "property": {
                                "type": "string",
                                "description": "e.g. 'soutien articulaire', 'action apaisante'",
                            },
                            "supporting_ingredient": {
                                "type": "string",
                                "description": "The ingredient that justifies this inference.",
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": (
                                    "high = well-documented mechanism; "
                                    "medium = commonly accepted; "
                                    "low = speculative or context-dependent."
                                ),
                            },
                        },
                    },
                },
                "text_field_mentions": {
                    "type": "array",
                    "description": (
                        "Formulation or functional property mentions found in free-text paragraph fields "
                        "(descriptions, arguments, conseils, en savoir plus, etc.). "
                        "Capture every occurrence — do not filter or summarize."
                    ),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["source_field", "excerpt", "detected_property", "already_in_structured_data"],
                        "properties": {
                            "source_field": {
                                "type": "string",
                                "description": "Name of the field where the mention was found.",
                            },
                            "excerpt": {
                                "type": "string",
                                "description": "Verbatim quote (max ~120 chars) containing the mention.",
                            },
                            "detected_property": {
                                "type": "string",
                                "description": "The formulation fact or functional property the excerpt describes.",
                            },
                            "already_in_structured_data": {
                                "type": "boolean",
                                "description": (
                                    "True if this property is already covered by parsed_active_ingredients "
                                    "or inferred_functional_properties. False = new information found only in text."
                                ),
                            },
                        },
                    },
                },
                "unsupported_claims_risk": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Properties or benefits that appear nowhere in the formulation data "
                        "and cannot be legitimately claimed. "
                        "Generation must never introduce these."
                    ),
                },
            },
        },
        "efficacy_claims": {
            "type": "object",
            "additionalProperties": False,
            "required": ["explicit_claims", "implicit_claims", "quantified_claims", "claim_language_markers"],
            "properties": {
                "explicit_claims": {
                    "type": "array",
                    "description": "Direct efficacy statements found verbatim in the product data.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["claim", "source_field", "strength"],
                        "properties": {
                            "claim": {"type": "string"},
                            "source_field": {
                                "type": "string",
                                "description": "Field name and whether origin is manufacturer or user experience.",
                            },
                            "strength": {
                                "type": "string",
                                "enum": ["strong", "moderate", "weak"],
                            },
                        },
                    },
                },
                "implicit_claims": {
                    "type": "array",
                    "description": "Efficacy suggested indirectly via product name, ingredient context, or positioning.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["description", "reasoning"],
                        "properties": {
                            "description": {"type": "string"},
                            "reasoning": {"type": "string"},
                        },
                    },
                },
                "quantified_claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any claim with a number, %, duration, or measurable outcome.",
                },
                "claim_language_markers": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["proof_language", "speed_language", "strength_language", "outcome_language", "hedging_language"],
                    "description": "Words/expressions signalling efficacy level, across all fields.",
                    "properties": {
                        "proof_language": {"type": "array", "items": {"type": "string"}},
                        "speed_language": {"type": "array", "items": {"type": "string"}},
                        "strength_language": {"type": "array", "items": {"type": "string"}},
                        "outcome_language": {"type": "array", "items": {"type": "string"}},
                        "hedging_language": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
        "scientific_evidence": {
            "type": "object",
            "additionalProperties": False,
            "required": ["studies_mentioned", "proof_assertions", "evidence_quality_summary"],
            "properties": {
                "studies_mentioned": {
                    "type": "array",
                    "description": "Every mention of a study, trial, test, or research found in the data.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "verifiability", "description"],
                        "properties": {
                            "name": {"type": "string"},
                            "verifiability": {
                                "type": "string",
                                "enum": ["verifiable", "partially_verifiable", "unverifiable"],
                            },
                            "description": {"type": "string"},
                        },
                    },
                },
                "proof_assertions": {
                    "type": "array",
                    "description": "Proof claims made without citing a specific study.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["assertion", "status"],
                        "properties": {
                            "assertion": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["substantiated", "partially_substantiated", "unsubstantiated"],
                            },
                        },
                    },
                },
                "evidence_quality_summary": {
                    "type": "string",
                    "description": "1–3 sentence direct summary for the generator on what can and cannot be claimed.",
                },
            },
        },
        "copy_angles": {
            "type": "array",
            "description": (
                "2–3 ready-to-use copy angles derived from formulation + tone_angle + primary_benefit. "
                "Written in copywriter language, not analyst language. "
                "The generator picks the angle matching the channel's reader portrait."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["angle", "hook", "grounded_in", "tone"],
                "properties": {
                    "angle": {
                        "type": "string",
                        "description": "Short name for the angle (e.g. 'Le chien qui vieillit bien').",
                    },
                    "hook": {
                        "type": "string",
                        "description": "Opening sentence written as the reader would feel it — not a product claim.",
                    },
                    "grounded_in": {
                        "type": "string",
                        "description": "The brief fact or formulation element that justifies this angle.",
                    },
                    "tone": {
                        "type": "string",
                        "enum": ["clinical", "reassuring", "educational", "conversion"],
                        "description": "Recommended tone for this angle.",
                    },
                },
            },
        },
        "user_feedback": {
            "type": "object",
            "additionalProperties": False,
            "required": ["found", "entries", "sentiment_summary"],
            "properties": {
                "found": {
                    "type": "boolean",
                    "description": "True if genuine user reviews or testimonials were found in the data.",
                },
                "entries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Verbatim or close-paraphrase of each genuine user feedback entry.",
                },
                "sentiment_summary": {
                    "type": "string",
                    "description": "Overall sentiment, or 'Aucun retour utilisateur trouvé dans les données produit.'",
                },
            },
        },
    },
}

_COMPLIANCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "allowed_claims",
        "forbidden_claims",
        "mandatory_mentions",
        "species_restrictions",
        "regulatory_notes",
        "efficacy_claim_compliance",
    ],
    "properties": {
        "allowed_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Safe claim formulations (e.g. 'soutient la mobilité', 'aide à réduire le stress').",
        },
        "forbidden_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Forbidden formulations including guérit, traite, élimine, 100%, garanti, etc.",
        },
        "mandatory_mentions": {
            "type": "string",
            "description": "Short paragraph (max 3 sentences) synthesizing the most important warnings, contraindications, and legal notices — only those with real safety or legal copy implications.",
        },
        "species_restrictions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "All species, age, weight, and condition restrictions — complete and verbatim, nothing omitted.",
        },
        "regulatory_notes": {
            "type": "string",
            "description": "Product regulatory category and copy implications.",
        },
        "efficacy_claim_compliance": {
            "type": "object",
            "additionalProperties": False,
            "required": ["safe_efficacy_formulations", "forbidden_efficacy_formulations", "study_citation_rules"],
            "description": "Efficacy claim rules derived from the evidence quality assessed in Step 2.",
            "properties": {
                "safe_efficacy_formulations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Efficacy formulations permitted given the available evidence level.",
                },
                "forbidden_efficacy_formulations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Efficacy formulations forbidden due to insufficient evidence, with reason.",
                },
                "study_citation_rules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Precise rules for citing studies in copy.",
                },
            },
        },
    },
}

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["brief", "compliance"],
    "properties": {
        "brief": _BRIEF_SCHEMA,
        "compliance": _COMPLIANCE_SCHEMA,
    },
}


def analyze(product: dict[str, Any], cfg: PipelineConfig) -> dict[str, Any]:
    """
    Single LLM call combining product intelligence brief + compliance boundary check.
    Returns {"brief": {...}, "compliance": {...}}.
    """
    with cfg.analyze_specs_path.open("r", encoding="utf-8") as f:
        specs = json.load(f)
    prompt = build_prompt(json.dumps(product, ensure_ascii=False, indent=2), specs)
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
    cfg.save_prompt_debug("analyze", {"model": cfg.reasoning_model, "temperature": 0.1, "messages": messages})

    client = OpenAI()
    response = client.responses.create(
        model=cfg.reasoning_model,
        input=messages,
        temperature=0.1,
        text={"format": {"type": "json_schema", "name": "product_analysis", "schema": _SCHEMA, "strict": True}},
    )
    return json.loads(response.output_text)
