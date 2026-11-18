"""Configuration: single authoritative definition and validation.

Only fields that some module actually uses are defined.  Unknown keys in a
YAML file raise ``ConfigError`` (no silent typos).  Fields are added here
when the module that consumes them is written.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigError(ValueError):
    """Invalid or unknown configuration."""


@dataclass
class InputConfig:
    xtekct: Optional[str] = None
    angle_file: Optional[str] = None          # *_ctdata.txt or *.ang
    projection_dir: Optional[str] = None      # supplied later by the user
    id_regex: str = r"(\d+)(?=\.[A-Za-z0-9]+$)"
    id_offset: int = 0                        # file_id + offset == angle id

    def validate(self) -> None:
        import re
        try:
            rx = re.compile(self.id_regex)
        except re.error as exc:
            raise ConfigError(f"input.id_regex invalid: {exc}") from exc
        if rx.groups < 1:
            raise ConfigError("input.id_regex needs one capture group")


@dataclass
class MemoryConfig:
    # Default is deliberately below the ~5.9 GiB seen free on the target
    # workstation (free -h), leaving headroom for the OS.
    limit_gib: float = 4.0

    def validate(self) -> None:
        if not self.limit_gib > 0.25:
            raise ConfigError("memory.limit_gib must be > 0.25")


@dataclass
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def validate(self) -> "Config":
        self.input.validate()
        self.memory.validate()
        return self


def _build(cls, data: Dict[str, Any], path: str = ""):
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'} must be a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"Unknown config key(s) at {path or 'top'}: "
                          f"{sorted(unknown)}")
    kwargs = {}
    for name, val in data.items():
        sub = getattr(cls(), name)
        kwargs[name] = _build(type(sub), val, f"{path}{name}.") \
            if is_dataclass(sub) else val
    return cls(**kwargs)


def load_config(path=None) -> Config:
    """Load YAML (or defaults if ``path`` is None) and validate."""
    if path is None:
        return Config().validate()
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read config {p}: {exc}") from exc
    return _build(Config, data).validate()
