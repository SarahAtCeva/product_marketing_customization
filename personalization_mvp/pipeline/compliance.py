from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import PipelineConfig


_SYSTEM = (
    "You are a French veterinary regulatory compliance expert. "
    "Your role is to define clear boundaries for what can and cannot be claimed "
    "about a pet health product in French e-commerce copy. "
    "Apply French and EU regulations for pet food supplements, veterinary medicines, "
    "and zootechnical products. Be strict. Protect against therapeutic overclaims."
)

_SCHEMA = {
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
            "description": "Warnings or legal mentions that must be included verbatim when relevant.",
        },
        "species_restrictions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Species-specific usage restrictions (e.g. 'Ne pas utiliser chez le chat de moins de 6 mois').",
        },
        "regulatory_notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Regulatory context relevant to copywriting (product category, authorization status, etc.).",
        },
    },
}


def check_compliance(
    product: dict[str, Any], brief: dict[str, Any], cfg: PipelineConfig
) -> dict[str, Any]:
    prompt = (
        "Pour ce produit de santé animale, définis les limites de communication conformes "
        "pour la rédaction de fiches produit e-commerce en français.\n\n"
        "Règles générales de base:\n"
        "- INTERDIT: 'guérit', 'traite', 'élimine', '100%', 'garanti', 'immédiat', "
        "'miracle', 'sans risque', 'cliniquement prouvé' (sauf certification vérifiable)\n"
        "- AUTORISÉ: 'soutient', 'aide à', 'contribue à', 'favorise', 'peut aider', "
        "'conçu pour', 'formulé pour'\n"
        "- Les contre-indications et avertissements du produit doivent être reproduits fidèlement.\n"
        "- Tenir compte de la catégorie réglementaire du produit "
        "(médicament vétérinaire, complément alimentaire, aliment diététique, etc.)\n\n"
        f"Brief produit:\n{json.dumps(brief, ensure_ascii=False, indent=2)}\n\n"
        f"Données brutes produit (pour les mentions légales et contre-indications):\n"
        f"{json.dumps(product, ensure_ascii=False, indent=2)}\n"
    )

    client = OpenAI()
    response = client.responses.create(
        model=cfg.reasoning_model,
        input=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.1,
        text={"format": {"type": "json_schema", "name": "compliance_check", "schema": _SCHEMA, "strict": True}},
    )
    return json.loads(response.output_text)
