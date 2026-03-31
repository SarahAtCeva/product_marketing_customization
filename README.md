# Product Marketing Customization
Ce pipeline a pour but de prendre en entrée des données produits + proposition de champs marketing et de générer un output contenant la personalisation de ces champs , tout en restant compliance-aware et en boostant l'adaptation à l'audience marketing . Il est basé sur des LLMs.

## Comment ça marche ?

Le système est un enchainement de 6 étapes expliquées comme :

1. **Extract** — Extraction des informations d'un produit depuis la matrice produit (EAN, brand, species, existing descriptions)
2. **Analyze** — Construction d'une contextualisation du produit et analyse des limites compliance relatives en amont (what claims are legally permissible)
3. **Generate** — Génération des descriptions et des champs (SEO) conditionés sur le canal de vente  + le profil d'expertise des utilisateurs.
4. **Validate** — Vérification des outputs de la génération par rapport aux spécifités de longueur max autorisée , de règles de Compliance + autres règles de rédactions préférables
5. **Judge** — Evaluation de la qualité en utilisant llm as a judge (alignment, compliance, formatting scores)
6. **Refine** — Intégration de l'évaluation de l'étape précédente pour une regénération afin d'améliorer la sortie finale.

Output is a set of ready-to-use marketing fields for a given channel/expertise combination.

## Channels and expertise profiles

**Channels** (defined in `channel_tone_specifications.json`):
- `animal_health_reseller` — generic veterinary resellers
- `la_compagnie_des_animaux` — La Compagnie des Animaux marketplace
- `amazon_animal_health` — Amazon Animal Health storefront

**Expertise profiles**:
- `veterinary_expert` — clinical, precise, technical language
- `retail_educator` — accessible, educational, reassuring
- `conversion_copywriter` — benefit-driven, persuasive

## Project structure

```
product_marketing_customization/
└── personalization_mvp/
    ├── app.py                            # Streamlit web UI
    ├── cli.py                            # CLI entry point
    ├── requirements.txt
    ├── field_specifications.json         # Field constraints (char limits, required, rules)
    ├── channel_tone_specifications.json  # Channel + expertise definitions
    ├── descriptions_content/
    │   ├── description_specs.json        # Description generation rules
    │   └── examples/                     # Few-shot examples for LLM prompts
    ├── seo_fields/
    │   └── seo_fields_specs.json         # SEO field constraints
    ├── pipeline/                         # Core pipeline stages
    │   ├── runner.py                     # Orchestrator
    │   ├── extract.py
    │   ├── analyze.py
    │   ├── generate.py
    │   ├── validate.py
    │   ├── judge.py
    │   └── retry.py
    └── runs/                             # Output artifacts (gitignored)
```

## Setup

**Prerequisites**: Python 3.12+, an OpenAI API key

```bash
cd personalization_mvp
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

## Usage

### Web UI (recommended)

```bash
cd personalization_mvp
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a CSV, select channel and expertise, run, then download the enriched output.

### CLI (single product)

```bash
cd personalization_mvp
python cli.py \
  --csv "Matrice produit IA(Feuil1) (2).csv" \
  --index 0 \
  --channel "animal_health_reseller" \
  --expertise "veterinary_expert" \
  --generation-model "gpt-4.1" \
  --reasoning-model "gpt-4.1-mini" \
  --audit-model "gpt-4.1-nano"
```

`--index` is the zero-based row index of the product in the CSV. Add `--prompt_debug` to save raw prompts alongside outputs.

### Jupyter (exploration)

```bash
cd personalization_mvp
source start_jupyter.sh
```

## CSV format

The input CSV must include at minimum:

| Column | Description |
|--------|-------------|
| `EAN` | Product barcode |
| `Marque` | Brand name |
| `Designation` | Product name/designation |

Additional columns (existing descriptions, species, warnings, etc.) are extracted automatically.

## Output

Each run writes a timestamped folder under `runs/local_YYYYMMDD_HHMMSS/`:

| File | Contents |
|------|----------|
| `input_product_index.json` | Parsed product data |
| `analysis_output.json` | Brief + compliance analysis |
| `generated_output.json` | Raw LLM-generated fields |
| `diff_output.json` | Before/after comparison |
| `judge_output.json` | Quality scores per field |
| `retry_output.json` | Final refined fields (ready for CSV merge) |

## Models

The pipeline uses three OpenAI models with different roles:

| Role | Default | Purpose |
|------|---------|---------|
| `--generation-model` | `gpt-4.1` | Content generation |
| `--reasoning-model` | `gpt-4.1-mini` | Analysis and judging |
| `--audit-model` | `gpt-4.1-nano` | Validation checks |
