from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    csv_path: Path
    index: int
    channel: str
    expertise: str
    generation_model: str   # descriptions (step 3) + retry (step 7)
    reasoning_model: str    # brief, compliance, SEO, judge (steps 1,2,4,6)
    audit_model: str        # consistency check (step 5)
    out_dir: Path
    field_specs_path: Path
    channel_tone_path: Path
    descriptions_specs_path: Path
    seo_specs_path: Path
    prompt_debug: bool = False
    enable_retry: bool = True

    def save_prompt_debug(self, name: str, data: dict[str, Any]) -> None:
        """Write a prompt record to prompts_debug/<name>.json when --prompt_debug is set."""
        if not self.prompt_debug:
            return
        debug_dir = self.out_dir / "prompts_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        with (debug_dir / f"{name}.json").open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def validate(self) -> None:
        """Fail fast before any LLM call: check env, files, and channel/expertise keys."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set in environment.")

        for attr in (
            "csv_path",
            "field_specs_path",
            "channel_tone_path",
            "descriptions_specs_path",
            "seo_specs_path",
        ):
            path: Path = getattr(self, attr)
            if not path.exists():
                raise FileNotFoundError(f"{attr} not found: {path}")

        with self.channel_tone_path.open("r", encoding="utf-8") as f:
            tone = json.load(f)

        channels = tone.get("channels") or {}
        if self.channel not in channels:
            raise ValueError(
                f"Channel '{self.channel}' not found in {self.channel_tone_path.name}. "
                f"Available: {sorted(channels)}"
            )

        profiles = tone.get("expertise_profiles") or {}
        if self.expertise not in profiles:
            raise ValueError(
                f"Expertise '{self.expertise}' not found in {self.channel_tone_path.name}. "
                f"Available: {sorted(profiles)}"
            )
