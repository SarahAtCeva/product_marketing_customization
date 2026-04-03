from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .analyze import analyze
from .config import PipelineConfig
from .extract import extract
from .generate import generate
from .judge import judge
from .retry import retry_fields
from .validate import compute_diff

EventCallback = Callable[[dict[str, Any]], None] | None


@dataclass
class RunResult:
    product: dict[str, Any]
    analysis: dict[str, Any]
    generated: dict[str, Any]
    diff: dict[str, Any]
    judgment: dict[str, Any]
    final: dict[str, Any]
    out_dir: Path


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _emit(cb: EventCallback, event: dict[str, Any]) -> None:
    if cb:
        cb(event)


def run_pipeline(cfg: PipelineConfig, on_event: EventCallback = None) -> RunResult:
    cfg.validate()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    _emit(on_event, {"type": "stage", "stage": "extract"})
    _emit(on_event, {"type": "log", "message": "[extract] Reading product from CSV..."})
    product = extract(cfg)
    _write(cfg.out_dir / "input_product_index.json", product)
    _emit(on_event, {"type": "log", "message": "[extract] Done."})

    _emit(on_event, {"type": "stage", "stage": "analyze"})
    _emit(on_event, {"type": "log", "message": "[analyze] Building product brief + compliance boundaries..."})
    analysis = analyze(product, cfg)
    _write(cfg.out_dir / "analysis_output.json", analysis)
    brief = analysis["brief"]
    compliance = analysis["compliance"]
    _emit(on_event, {"type": "log", "message": (
        f"[analyze] Benefit: {brief.get('primary_benefit', '?')} | "
        f"Tone: {brief.get('tone_angle', '?')} | "
        f"{len(compliance.get('allowed_claims', []))} allowed / "
        f"{len(compliance.get('forbidden_claims', []))} forbidden claims."
    )})

    _emit(on_event, {"type": "stage", "stage": "generate"})
    _emit(on_event, {"type": "log", "message": "[generate] Generating descriptions and SEO fields..."})
    generated = generate(product, cfg, brief, compliance)
    _write(cfg.out_dir / "generated_output.json", generated)
    _emit(on_event, {"type": "log", "message": "[generate] Done."})

    _emit(on_event, {"type": "stage", "stage": "diff"})
    _emit(on_event, {"type": "log", "message": "[diff] Computing diff vs original CSV values..."})
    with cfg.field_specs_path.open("r", encoding="utf-8") as f:
        field_specs = json.load(f)
    diff = compute_diff(product, generated["generated_fields"], field_specs)
    _write(cfg.out_dir / "diff_output.json", diff)
    _emit(on_event, {"type": "log", "message": f"[diff] {diff['summary']['changed_fields']} fields changed."})

    _emit(on_event, {"type": "stage", "stage": "judge"})
    _emit(on_event, {"type": "log", "message": "[judge] Running LLM quality assessment..."})
    judgment = judge(diff, cfg, brief, compliance)
    _write(cfg.out_dir / "judge_output.json", judgment)
    _emit(on_event, {"type": "log", "message": f"[judge] Grade: {judgment.get('channel_alignment_grade')}."})

    _emit(on_event, {"type": "stage", "stage": "retry"})
    if cfg.enable_retry:
        _emit(on_event, {"type": "log", "message": "[retry] Applying judge feedback to changed fields..."})
        final = retry_fields(diff, judgment, cfg, brief, compliance)
        _write(cfg.out_dir / "retry_output.json", final)
        _emit(on_event, {"type": "log", "message": "[retry] Done."})
    else:
        _emit(on_event, {"type": "log", "message": "[retry] Skipped (disabled)."})
        final = {"generated_fields": generated["generated_fields"]}
        _write(cfg.out_dir / "retry_output.json", final)

    artifacts: dict[str, str] = {
        "input": str(cfg.out_dir / "input_product_index.json"),
        "analysis": str(cfg.out_dir / "analysis_output.json"),
        "generated": str(cfg.out_dir / "generated_output.json"),
        "diff": str(cfg.out_dir / "diff_output.json"),
        "judge": str(cfg.out_dir / "judge_output.json"),
        "retry": str(cfg.out_dir / "retry_output.json"),
    }
    if cfg.prompt_debug:
        artifacts["prompts_debug"] = str(cfg.out_dir / "prompts_debug")

    manifest = {
        "pipeline_version": "mvp_v1",
        "source": {
            "csv": str(cfg.csv_path),
            "index": cfg.index,
            "channel": cfg.channel,
            "expertise": cfg.expertise,
            "generation_model": cfg.generation_model,
            "reasoning_model": cfg.reasoning_model,
            "audit_model": cfg.audit_model,
        },
        "artifacts": artifacts,
    }
    _write(cfg.out_dir / "pipeline_manifest.json", manifest)

    _emit(on_event, {"type": "complete", "artifacts": manifest["artifacts"]})

    return RunResult(
        product=product,
        analysis=analysis,
        generated=generated,
        diff=diff,
        judgment=judgment,
        final=final,
        out_dir=cfg.out_dir,
    )
