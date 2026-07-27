"""Regression tests for DGST source-tau / transport-top-k config plumbing."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERIC_SWEEP_KEYS = {
    "four_gate_source_tau_values",
    "four_gate_transport_top_k_values",
    "four_gate_capped_topmass_alphas",
}
DIRECT_SWEEP_KEYS = {
    "source_tau_values",
    "transport_top_k_values",
    "capped_topmass_alphas",
}


def _calls(path: Path, function_name: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if isinstance(node.func, ast.Name):
            called = node.func.id
        if called == function_name:
            matches.append(node)
    return matches


def _keyword_names(call: ast.Call) -> set[str]:
    return {keyword.arg for keyword in call.keywords if keyword.arg is not None}


def test_all_model_batch_paths_forward_sweep_lists() -> None:
    wrapper_paths = [
        ROOT / "models" / "internvl_wrapper.py",
        ROOT / "models" / "llava_wrapper.py",
        ROOT / "models" / "qwen_wrapper.py",
        ROOT / "models" / "qwen3_vl_wrapper.py",
        ROOT / "models" / "llava_onevision_wrapper.py",
    ]
    for path in wrapper_paths:
        calls = _calls(path, "compute_dgst_t_batch_from_captures")
        assert calls, f"No DGST batch call found in {path.name}"
        for call in calls:
            missing = GENERIC_SWEEP_KEYS - _keyword_names(call)
            assert not missing, f"{path.name} does not forward {sorted(missing)}"


def test_direct_four_gate_paths_forward_sweep_lists() -> None:
    for filename in ("internvl_wrapper.py", "llava_wrapper.py"):
        path = ROOT / "models" / filename
        calls = _calls(path, "compute_four_gate_dgst_batch_from_captures")
        assert calls, f"No direct four-gate call found in {filename}"
        for call in calls:
            missing = DIRECT_SWEEP_KEYS - _keyword_names(call)
            assert not missing, f"{filename} does not forward {sorted(missing)}"


def test_prompt_target_options_forward_sweep_lists() -> None:
    path = ROOT / "models" / "prompt_target.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_dgst_options"
    )
    returned_dict = next(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    )
    option_keys = {
        key.value
        for key in returned_dict.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    missing = GENERIC_SWEEP_KEYS - option_keys
    assert not missing, f"prompt_target._dgst_options omits {sorted(missing)}"
