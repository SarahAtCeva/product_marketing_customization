from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import PipelineConfig


_SYSTEM = (
    "You are a veterinary product analyst specializing in French pet health e-commerce. "
    "Your role is to deeply understand a product before any copy is written. "
    "Be precise, grounded in the provided data only, and flag gaps honestly. "
    "Never invent benefits, ingredients, certifications, or efficacy claims."
)

_SCHEMA = {
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
            "description": (
                "Questions a potential buyer would ask about this product, ordered by priority "
                "(most important first). Each question is flagged as answered or not by the "
                "existing product data."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["priority", "question", "answered", "answer_summary"],
                "properties": {
                    "priority": {
                        "type": "integer",
                        "description": "1 = most important buyer concern, higher = less critical.",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question as a buyer would phrase it, in French.",
                    },
                    "answered": {
                        "type": "boolean",
                        "description": "True if the existing product data answers this question.",
                    },
                    "answer_summary": {
                        "type": "string",
                        "description": (
                            "If answered: short summary of the answer from product data. "
                            "If not answered: 'Information absente des données produit.'"
                        ),
                    },
                },
            },
        },
    },
}


def build_brief(product: dict[str, Any], cfg: PipelineConfig) -> dict[str, Any]:
    prompt = (
        "Analyse ce produit de santé animale et produis un brief structuré.\n\n"
        "Règles strictes:\n"
        "- Base-toi uniquement sur les données fournies.\n"
        "- Si une information est absente ou trop vague, note-la dans data_gaps.\n"
        "- Ne pas inventer de bénéfices, ingrédients, certifications ou durées.\n"
        "- Pour use_cases: pense aux situations concrètes d'un propriétaire d'animal.\n\n"
        "Pour buyer_questions:\n"
        "- Génère 6 à 10 questions qu'un acheteur potentiel poserait sur ce produit.\n"
        "- Ordonne-les par priorité décroissante (1 = préoccupation principale de l'acheteur).\n"
        "- Pense aux questions sur: efficacité, sécurité, mode d'emploi, composition, "
        "espèce/âge ciblé, durée du traitement, compatibilité avec d'autres produits, prix/format.\n"
        "- Pour chaque question, vérifie honnêtement si les données produit y répondent.\n"
        "- answered=true uniquement si la réponse est clairement présente dans les données.\n"
        "- answer_summary doit être court (1 phrase max).\n\n"
        f"Données produit:\n{json.dumps(product, ensure_ascii=False, indent=2)}\n"
    )

    client = OpenAI()
    response = client.responses.create(
        model=cfg.reasoning_model,
        input=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.1,
        text={"format": {"type": "json_schema", "name": "product_brief", "schema": _SCHEMA, "strict": True}},
    )
    return json.loads(response.output_text)
