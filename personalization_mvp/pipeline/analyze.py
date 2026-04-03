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


def build_prompt(product_json: str) -> str:
    return (
        "Analyse le produit de santé animale ci-dessous. Suis les 4 étapes dans l'ordre strict.\n"
        "Chaque étape alimente la suivante — ne saute rien.\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ÉTAPE 1 · ANALYSE DE LA FORMULATION\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Objectif : construire une carte complète de ce que le produit CONTIENT et de ce que "
        "sa formulation PERMET LÉGITIMEMENT de revendiquer.\n\n"

        "A) CHAMPS STRUCTURÉS (Composition, Additifs nutritionnels, Constituants analytiques)\n"
        "   → parsed_active_ingredients : un ingrédient = une entrée. "
        "Inclus le dosage exact s'il est présent, sinon 'non précisé'.\n"
        "   → inferred_functional_properties : pour chaque ingrédient, dérive les propriétés "
        "fonctionnelles DOCUMENTÉES (pas spéculatives). "
        "Cite l'ingrédient qui justifie chaque propriété. "
        "Utilise un langage orienté bénéfice concret, pas des labels génériques.\n"
        "     Exemple BON  : 'soutien de la fonction articulaire chez le chien âgé' (glucosamine, high)\n"
        "     Exemple FAIBLE : 'action apaisante' (sans préciser sur quoi, ni quel ingrédient)\n\n"

        "B) CHAMPS TEXTE — scanne EXHAUSTIVEMENT tous les champs paragraphe :\n"
        "   Désignation, Description courte, Description longue, Description 400 car., "
        "Description 600 car., En savoir plus, Argu 1–5, Conseils d'utilisation, "
        "Titre produit, et tout autre champ textuel présent.\n"
        "   → text_field_mentions : une entrée par mention trouvée. Ne filtre pas, ne résume pas. "
        "Indique si l'info est déjà couverte par les champs structurés (already_in_structured_data).\n"
        "   ATTENTION : les meilleurs arguments de vente sont souvent cachés dans des champs "
        "secondaires. Ne les rate pas.\n\n"

        "C) → unsupported_claims_risk : liste les propriétés/bénéfices qui n'apparaissent "
        "NI dans la formulation NI dans aucun texte. "
        "Le générateur ne devra JAMAIS les mentionner.\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ÉTAPE 2 · CLAIMS D'EFFICACITÉ, PREUVES & RETOURS UTILISATEURS\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Objectif : inventorier TOUT ce que les données disent sur l'efficacité du produit, "
        "avec quel niveau de preuve, pour que le générateur sache exactement "
        "ce qu'il peut affirmer et comment le formuler.\n\n"

        "A) efficacy_claims\n"
        "   • explicit_claims : toute affirmation directe d'efficacité, verbatim. "
        "Classe par strength (strong/moderate/weak). "
        "Sépare mentalement les claims du FABRICANT vs ceux rapportant une EXPÉRIENCE UTILISATEUR, "
        "et reflète cette distinction dans le champ source_field.\n"
        "   • implicit_claims : efficacité suggérée indirectement (nom du produit évocateur, "
        "mention d'ingrédient dans un contexte d'efficacité, positionnement marketing). "
        "Explique ton raisonnement dans 'reasoning'.\n"
        "   • quantified_claims : tout claim avec chiffre, %, durée, résultat mesurable. "
        "Ce sont les plus puissants en copy — sois exhaustif.\n"
        "   • claim_language_markers : chaque mot/expression signalant un niveau d'efficacité, "
        "dans TOUS les champs. Catégorise : proof_language, speed_language, "
        "strength_language, outcome_language, hedging_language.\n\n"

        "B) scientific_evidence\n"
        "   • studies_mentioned : TOUTE mention d'étude, essai, test, recherche, validation. "
        "Sois particulièrement attentif à distinguer :\n"
        "     - étude nommée et vérifiable (is_named=true, verifiable)\n"
        "     - référence institutionnelle sans étude précise (partially_verifiable)\n"
        "     - assertion générique 'cliniquement prouvé' sans source (unverifiable)\n"
        "   • proof_assertions : affirmations de preuve SANS étude citée. "
        "Évalue si elles sont substantiated, partially_substantiated, ou unsubstantiated "
        "en croisant avec studies_mentioned.\n"
        "   • evidence_quality_summary : 1–3 phrases. Sois direct et utile pour le générateur. "
        "Exemple : 'Une seule étude terrain nommée (Étude X, 2021) portant sur la palatabilité. "
        "Aucune preuve clinique d'efficacité thérapeutique. "
        "Le copy peut citer le taux de satisfaction mais pas revendiquer une action médicale.'\n\n"

        "C) user_feedback\n"
        "   • Scanne TOUS les champs texte pour des avis, témoignages, notes, retours clients.\n"
        "   • ATTENTION : ne confonds PAS le discours marketing du fabricant avec de vrais retours "
        "utilisateurs. 'Nos clients adorent ce produit' dans un Argu ≠ un vrai témoignage.\n"
        "   • Si aucun retour trouvé : found=false, entries=[], "
        "sentiment_summary='Aucun retour utilisateur trouvé dans les données produit.'\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ÉTAPE 3 · BRIEF CRÉATIF PRODUIT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Objectif : fournir au générateur une vision claire, actionnable et orientée conversion "
        "du produit.\n\n"

        "• product_summary : 1–2 phrases factuelles. Dis ce que c'est, pour qui, et ce que ça fait. "
        "Pas de flou marketing.\n"
        "  BON  : 'Complément alimentaire en comprimés pour chiens adultes, à base de glucosamine "
        "et chondroïtine, formulé pour soutenir la mobilité articulaire.'\n"
        "  FAIBLE : 'Produit innovant pour le bien-être de votre compagnon.'\n\n"

        "• target_species, target_profile : sois aussi spécifique que les données le permettent. "
        "'Chien adulte de grande race (>25 kg) à activité physique soutenue' "
        "> 'Chien'.\n\n"

        "• primary_benefit : le bénéfice n°1, formulé comme un pet owner le ressentirait. "
        "Ancré dans les résultats de l'Étape 1 — pas inventé.\n"
        "• secondary_benefits : bénéfices additionnels, chacun en phrase courte et concrète.\n\n"

        "• use_cases : 4–6 situations CONCRÈTES de la vie d'un propriétaire d'animal. "
        "Pense aux déclencheurs d'achat réels :\n"
        "  - Le vétérinaire a recommandé un soutien articulaire après un bilan\n"
        "  - Le chien montre des signes de raideur au lever le matin\n"
        "  - Transition alimentaire après une sensibilité digestive\n"
        "  PAS : 'Pour les chiens qui ont besoin de soutien' (trop vague)\n\n"

        "• key_differentiator : ce qui distingue CE produit de sa catégorie générique. "
        "Si rien de distinctif n'apparaît dans les données, écris-le honnêtement.\n\n"

        "• tone_angle : choisis parmi clinical / reassuring / educational / conversion "
        "et justifie brièvement dans la valeur (ex : 'reassuring — produit post-chirurgie, "
        "le propriétaire cherche une solution fiable et douce').\n\n"

        "• existing_arguments : copie VERBATIM le contenu des champs Argu 1–5 présents "
        "dans les données. Ignore les champs vides.\n\n"

        "• explanation_of_the_concept_innovation : si le produit repose sur une idée médicale "
        "ou une innovation (technologie de libération, brevet, formule exclusive…), "
        "explique-la clairement. Sinon, indique 'Aucune innovation spécifique identifiée.'\n\n"

        "• data_gaps : liste TOUTES les informations manquantes ou trop vagues. "
        "Le générateur utilisera cette liste pour éviter d'inventer. "
        "Sois spécifique : 'Dosage de la glucosamine non précisé' "
        "> 'Informations manquantes sur la composition'.\n\n"

        "• buyer_questions : 6–10 questions prioritaires qu'un acheteur poserait VRAIMENT. "
        "Pense au parcours d'achat :\n"
        "  - Phase recherche : 'Est-ce adapté à mon chien de 14 ans avec de l'arthrose ?'\n"
        "  - Phase comparaison : 'Quelle différence avec [catégorie concurrente] ?'\n"
        "  - Phase décision : 'Combien de temps avant de voir des résultats ?'\n"
        "  - Phase utilisation : 'Mon chien peut-il le prendre avec son traitement actuel ?'\n"
        "Pour chaque question : answered=true UNIQUEMENT si la réponse est clairement "
        "dans les données. Sinon answered=false et answer_summary='Information absente "
        "des données produit.'\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ÉTAPE 4 · CADRE DE CONFORMITÉ RÉGLEMENTAIRE\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Objectif : poser les garde-fous que le générateur ne pourra PAS franchir. "
        "Ce que tu autorises ici, il L'UTILISERA. Ce que tu interdis, il l'évitera. "
        "Sois précis.\n\n"

        "• allowed_claims : formulations SÛRES, prêtes à l'emploi. "
        "Préfère les tournures avec 'soutient', 'aide à', 'contribue à', 'favorise', "
        "'conçu pour', 'formulé pour'.\n"
        "• forbidden_claims : formulations INTERDITES. "
        "Inclus systématiquement : 'guérit', 'traite', 'élimine', '100%', 'garanti', "
        "'immédiat', 'miracle', 'sans risque'. "
        "Ajoute toute formulation spécifique au produit qui serait trompeuse.\n"
        "• mandatory_mentions : avertissements, contre-indications, mentions légales "
        "à reproduire VERBATIM.\n"
        "• species_restrictions : restrictions d'espèce, d'âge, de poids, d'état.\n"
        "• regulatory_notes : catégorie réglementaire du produit et implications "
        "pour le copy (ex : 'Aliment complémentaire — ne peut pas revendiquer "
        "d'action thérapeutique').\n\n"

        "• efficacy_claim_compliance :\n"
        "  → safe_efficacy_formulations : formulations d'efficacité autorisées "
        "VU LE NIVEAU DE PREUVE identifié en Étape 2. "
        "Sois cohérent : si aucune étude nommée n'existe, 'efficacité prouvée' "
        "ne peut PAS être safe.\n"
        "  → forbidden_efficacy_formulations : formulations interdites car "
        "le niveau de preuve est insuffisant. Sois explicite sur POURQUOI "
        "c'est interdit (aide le générateur à comprendre la logique).\n"
        "  → study_citation_rules : règles précises pour citer les études dans le copy. "
        "Exemples :\n"
        "    - 'L'étude X (2021) peut être citée avec le résultat Y.'\n"
        "    - 'Le test consommateur peut être cité comme donnée de satisfaction, "
        "PAS comme preuve clinique.'\n"
        "    - 'Aucune étude nommée → interdiction absolue de \"scientifiquement prouvé\" "
        "ou \"cliniquement testé\".'\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "DONNÉES PRODUIT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
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
            "type": "array",
            "items": {"type": "string"},
            "description": "Warnings, contraindications, and legal notices to reproduce verbatim.",
        },
        "species_restrictions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Species, age, weight, or condition restrictions.",
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
    prompt = build_prompt(json.dumps(product, ensure_ascii=False, indent=2))
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
