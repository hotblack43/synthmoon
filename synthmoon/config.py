from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]

    @property
    def utc(self) -> str:
        return str(self.raw["time"]["utc"])

    @property
    def observer_mode(self) -> str:
        return str(self.raw["observer"]["mode"])

    @property
    def observer(self) -> Dict[str, Any]:
        return dict(self.raw["observer"])

    @property
    def camera(self) -> Dict[str, Any]:
        return dict(self.raw["camera"])

    @property
    def paths(self) -> Dict[str, Any]:
        return dict(self.raw["paths"])

    @property
    def moon(self) -> Dict[str, Any]:
        return dict(self.raw["moon"])

    @property
    def earth(self) -> Dict[str, Any]:
        return dict(self.raw["earth"])

    @property
    def illumination(self) -> Dict[str, Any]:
        return dict(self.raw["illumination"])

    @property
    def shadows(self) -> Dict[str, Any]:
        return dict(self.raw["shadows"])

    @property
    def output(self) -> Dict[str, Any]:
        return dict(self.raw["output"])


def load_config(path: str | Path) -> Config:
    p = Path(path)
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    return Config(raw)
