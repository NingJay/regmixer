from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def build_base_control_state(workdir: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "workdir": workdir,
        "phases": {},
        "updated_at": time.time(),
    }


def read_control_state(path: str, workdir: str) -> Dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return build_base_control_state(workdir)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload.setdefault("schema_version", 1)
    payload.setdefault("workdir", workdir)
    payload.setdefault("phases", {})
    payload.setdefault("updated_at", time.time())
    return payload


def ensure_phase_state(state: Dict[str, Any], phase: str) -> Dict[str, Any]:
    phases = state.setdefault("phases", {})
    phase_state = phases.setdefault(phase, {})
    return phase_state


def write_control_state(path: str, state: Dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(state_path)
