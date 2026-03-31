from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from .config import PipelineConfig

_STATUS_MARKERS = {"TODO", "OK", "NON", "OUI"}


def _normalize(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").replace("\ufffd", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def _read_csv(path: Path) -> list[list[str]]:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.reader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Unable to decode CSV with utf-8-sig or latin-1")


def _find_header_row(rows: list[list[str]]) -> int:
    for i, row in enumerate(rows[:10]):
        cols = [_normalize(c) for c in row]
        if "ean" in cols and "marque" in cols and "designation" in cols:
            return i
    raise ValueError(
        "Could not detect CSV header row in first 10 rows "
        "(expected columns: EAN, Marque, Designation)."
    )


def _find_status_row(rows: list[list[str]], start: int) -> int | None:
    for i in range(start, len(rows)):
        values = {c.strip().upper() for c in rows[i] if c.strip()}
        if "TODO" in values and values & _STATUS_MARKERS:
            return i
    return None


def _header_index(header: list[str]) -> dict[str, int]:
    return {_normalize(name): i for i, name in enumerate(header)}


def _pick(row: list[str], idx: dict[str, int], aliases: list[str]) -> str | None:
    for alias in aliases:
        col = idx.get(_normalize(alias))
        if col is not None:
            val = row[col].strip() if col < len(row) else ""
            return val or None
    return None


def _block(
    row: list[str], idx: dict[str, int], spec: dict[str, list[str]]
) -> dict[str, str | None]:
    return {key: _pick(row, idx, aliases) for key, aliases in spec.items()}


def extract(cfg: PipelineConfig) -> dict[str, Any]:
    rows = _read_csv(cfg.csv_path)
    if not rows:
        raise ValueError("CSV is empty")

    header_idx = _find_header_row(rows)
    header = rows[header_idx]
    idx = _header_index(header)

    status_idx = _find_status_row(rows, header_idx + 1)
    end = status_idx if status_idx is not None else len(rows)

    data_rows = [
        (r + [""] * max(0, len(header) - len(r)))
        for r in rows[header_idx + 1 : end]
        if any(c.strip() for c in r)
    ]

    if cfg.index < 0 or cfg.index >= len(data_rows):
        raise IndexError(
            f"Product index {cfg.index} out of range. Available: 0..{len(data_rows) - 1}"
        )
    row = data_rows[cfg.index]

    core_spec = {
        "EAN": ["EAN"],
        "SKU": ["SKU", " SKU "],
        "ASIN": ["ASIN", " ASIN "],
        "Marque": ["Marque"],
        "Gamme": ["Gamme"],
        "Designation": ["Designation"],
        "Présentation": ["Présentation"],
        "Espèce": ["Espèce"],
        "Taille de l'animal": ["Taille de l'animal", " Taille de l'animal "],
        "Catégorie (Clients)": ["Catégorie (Clients)", " Catégorie (Clients) "],
        "Catégorisation": ["Catégorisation", " Catégorisation "],
    }
    commercial_spec = {
        "Titre produit": ["Titre produit"],
        "Titre court": ["Titre court du produit (60 caractères max)\nEviter de mettre la contenance"],
        "Slug": ["Slug"],
        "META Title": ["META Title", "META Title "],
        "META Description": ["META Description"],
        "Mots clés": ["Mots clés"],
        "Argu 1": ["Argu 1"],
        "Argu 2": ["Argu 2"],
        "Argu 3": ["Argu 3"],
        "Argu 4": ["Argu 4"],
        "Argu 5": ["Argu 5"],
        "Cross-sell1": ["Cross-sell1 = produit apparenté ou faisant partie de la routine, elle apparaît dans les produits complémentaires dans a fiche produit (mettre l'EAN)*"],
        "Cross-sell2": ["Cross-sell2 (mettre l'EAN)*"],
        "Cross-sell3": ["Cross-sell3 (mettre l'EAN)*"],
        "Cross-sell4": ["Cross-sell4 (mettre l'EAN)*"],
    }
    scientific_spec = {
        "Additifs nutritionnels": ["Additifs nutritionnels"],
        "Composition": ["Composition"],
        "Contre-indication, avertissements, mentions obligatoires": [
            "Contre-indication, avertissements, mentions obligatoires\ufffd",
            "Contre-indication, avertissements, mentions obligatoires",
        ],
        "Conseils d'utilisation institutionnels": ["Conseils d'utilisation institutionnels"],
        "Certification": ["Certification"],
        "Pays d'origine": ["Pays d'origine"],
        "Produit bio ?": ["Produit bio ?", "Produit bio ? "],
        "Inflammable ?": ["Inflammable ? (Oui/Non)"],
        "Produit frais": ["Produit frais", "Produit frais "],
    }
    content_spec = {
        "Description courte (175 caractères)": [
            "Description courte\n (Courte, rapide, elle apparait dans l'aperçu lors d'une recherche sur le site et dans les méta ne doit pas, elle ne doit pas dépasser 175 caractères)"
        ],
        "Description du produit unique/site 400 caractères": [
            "Description du produit unique/site\n400 caractères",
            "Description du produit unique/site\n400 caractères ",
        ],
        "Description courte 600 caractères": ["Description courte 600 caractères"],
        "Description longue": ["Description longue"],
        "En savoir plus": ['"""En savoir plus"""'],
        "Conseils d'utilisation uniques (SEO)": ["Conseils d'utilisation uniques (SEO)"],
    }

    return {
        "source": {
            "csv_file": str(cfg.csv_path),
            "header_row_index_0_based": header_idx,
            "status_row_index_0_based": status_idx,
            "product_index_0_based": cfg.index,
        },
        "core_identity": _block(row, idx, core_spec),
        "commercial_seo_inputs": _block(row, idx, commercial_spec),
        "scientific_regulatory_inputs": _block(row, idx, scientific_spec),
        "content_body_inputs": _block(row, idx, content_spec),
    }
