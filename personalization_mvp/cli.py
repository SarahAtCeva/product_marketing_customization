#!/usr/bin/env python3
"""Thin CLI wrapper — calls run_pipeline() directly, no subprocess."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline import PipelineConfig, run_pipeline

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Product personalization pipeline (MVP)")
    p.add_argument("--csv", required=True, help="Path to source CSV")
    p.add_argument("--index", type=int, required=True, help="0-based product index")
    p.add_argument("--channel", required=True, help="Channel key (must exist in channel_tone_specifications.json)")
    p.add_argument("--expertise", required=True, help="Expertise key (must exist in channel_tone_specifications.json)")
    p.add_argument("--generation-model", default="gpt-4.1", help="Model for descriptions + retry (default: gpt-4.1)")
    p.add_argument("--reasoning-model", default="gpt-4.1-mini", help="Model for SEO + judge (default: gpt-4.1-mini)")
    p.add_argument("--audit-model", default="gpt-4.1-nano", help="Model for consistency check (default: gpt-4.1-nano)")
    p.add_argument("--out-dir", default=None, help="Output directory (default: runs/local_TIMESTAMP)")
    p.add_argument("--field-specifications", default=str(ROOT / "field_specifications.json"))
    p.add_argument("--channel-tone", default=str(ROOT / "channel_tone_specifications.json"))
    p.add_argument("--description-specs", default=str(ROOT / "descriptions_content" / "description_specs.json"))
    p.add_argument("--seo-specs", default=str(ROOT / "seo_fields" / "seo_fields_specs.json"))
    p.add_argument("--prompt_debug", action="store_true", help="Save all LLM prompts to prompts_debug/ in the run directory")
    return p.parse_args()


def _resolve(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _log(event: dict[str, Any]) -> None:
    typ = event.get("type")
    if typ == "stage":
        print(f"\n==> Stage: {event['stage'].upper()}")
    elif typ == "log":
        print(event["message"])
    elif typ == "complete":
        print("\nPipeline complete. Artifacts:")
        for name, path in event.get("artifacts", {}).items():
            print(f"  {name}: {path}")
    elif typ == "error":
        print(f"ERROR: {event['message']}")


def main() -> None:
    args = parse_args()
    out_dir = (
        _resolve(args.out_dir)
        if args.out_dir
        else ROOT / "runs" / datetime.now().strftime("local_%Y%m%d_%H%M%S")
    )

    cfg = PipelineConfig(
        csv_path=_resolve(args.csv),
        index=args.index,
        channel=args.channel,
        expertise=args.expertise,
        generation_model=args.generation_model,
        reasoning_model=args.reasoning_model,
        audit_model=args.audit_model,
        out_dir=out_dir,
        field_specs_path=_resolve(args.field_specifications),
        channel_tone_path=_resolve(args.channel_tone),
        descriptions_specs_path=_resolve(args.description_specs),
        seo_specs_path=_resolve(args.seo_specs),
        prompt_debug=args.prompt_debug,
    )

    run_pipeline(cfg, on_event=_log)


if __name__ == "__main__":
    main()
