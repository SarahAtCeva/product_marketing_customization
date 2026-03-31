from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import PipelineConfig


_SYSTEM = (
    "You are a senior veterinary product analyst and French regulatory compliance expert "
    "specializing in pet health e-commerce. "
    "Your role is to deeply understand a product AND define the communication boundaries "
    "Be precise and grounded in the provided data only. "
    "Never invent benefits, ingredients, certifications, or efficacy claims."
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
        "formulation_analysis",
        "existing_arguments",
        "explanation_of_the_concept_innovation",
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
            "description": "the medical idea or innovation about the product if mentioned",
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
                                "description": "Name of the field where the mention was found (e.g. 'Description longue', 'Argu 1').",
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
            "description": "Claims that must never appear (e.g. 'guérit l\\'arthrose', 'traite les infections').",
        },
        "mandatory_mentions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Warnings or legal mentions that must be reproduced verbatim when relevant.",
        },
        "species_restrictions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Species-specific usage restrictions.",
        },
        "regulatory_notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Regulatory context relevant to copywriting (product category, authorization, etc.).",
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
    prompt = (
        "Analyse ce produit de santé animale en trois étapes dans l'ordre indiqué ci-dessous.\n\n"
        "== ÉTAPE 1 : ANALYSE FORMULATION (à compléter EN PREMIER) ==\n"
        "A) Champs structurés — parse 'Composition' ingrédient par ingrédient :\n"
        "   - Liste chaque ingrédient actif dans parsed_active_ingredients avec son champ source et sa quantité.\n"
        "   - Pour chaque ingrédient, dérive les propriétés fonctionnelles si documentées"
        "inferred_functional_properties (ex : glucosamine → soutien articulaire, confidence=high). "
        "N'accepte une propriété que si un ingrédient précis la justifie.\n"
        "B) Champs texte — scan TOUS les champs de type paragraphe "
        "(Désignation, Description courte, Description longue, Description 400 car., "
        "Description 600 car., En savoir plus, Argu 1-5, Conseils d'utilisation, Titre produit, etc.) :\n"
        "   - Repère toute mention d'un ingrédient, d'une formule, d'un mécanisme d'action ou d'une "
        "propriété fonctionnelle.\n"
        "   - Pour chaque mention, crée une entrée dans text_field_mentions avec : le champ source, "
        "un extrait verbatim (≤120 car.), la propriété détectée, et already_in_structured_data=true "
        "si cette propriété est déjà couverte par les champs structurés, false sinon.\n"
        "   - Ne filtre pas, ne résume pas : capture toutes les occurrences.\n"
        "C) Dans unsupported_claims_risk : liste les propriétés qui n'apparaissent NI dans la formulation "
        "NI dans les textes — ces claims sont interdits en génération.\n\n"
        "== ÉTAPE 2 : BRIEF PRODUIT ==\n"
        "- Dans existing_arguments : copie verbatim le contenu des champs Argu 1, Argu 2, Argu 3, Argu 4, Argu 5 "
        "présents dans les données produit (ignore les champs vides ou absents).\n"
        "- Base primary_benefit et secondary_benefits sur les résultats de l'étape 1 uniquement.\n"
        "- Si une information est absente ou trop vague, note-la dans data_gaps.\n"
        "- Ne pas inventer de bénéfices, ingrédients, certifications ou durées.\n"
        "- Pour use_cases : pense aux situations concrètes d'un propriétaire d'animal.\n"
        "- Pour buyer_questions : génère 6 à 10 questions prioritaires (1 = plus important), "
        "avec answered=true uniquement si la réponse est clairement dans les données.\n\n"
        "== ÉTAPE 3 : CONFORMITÉ ==\n"
        "- INTERDIT : 'guérit', 'traite', 'élimine', '100%', 'garanti', 'immédiat', "
        "'miracle', 'sans risque', 'cliniquement prouvé' (sauf certification vérifiable).\n"
        "- AUTORISÉ : 'soutient', 'aide à', 'contribue à', 'favorise', 'peut aider', "
        "'conçu pour', 'formulé pour'.\n"
        "- Reproduire fidèlement les contre-indications et avertissements du produit.\n"
        "- Tenir compte de la réglementation française et européenne "
        "(médicament vétérinaire, complément alimentaire, aliment diététique, etc.).\n\n"
        f"Données produit:\n{json.dumps(product, ensure_ascii=False, indent=2)}\n"
    )

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
