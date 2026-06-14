"""DGST-T continuous risk features.

This module keeps only the core DGST-T signals requested for the TGD port:
source_dist from FFN update, target_dist from attention times semantic
probability, Wasserstein/OT transport risk, prompt cosine features, and
context-confidence features.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import torch
import torch.nn.functional as F

EPS = 1e-12


def compute_dgst_t(
    dgst_t_raw: dict[str, Any],
    *,
    tau: float = 0.07,
    transport_top_k: int = 64,
    cost_mode: str = "direct",
    lambda_d: float = 1.0,
    lambda_s: float = 1.0,
    lambda_t: float = 1.0,
    lambda_int: float = 1.0,
    baseline_layers: int = 10,
    risk_start_layer: int = 15,
    alpha: float = 2.0,
    ot_solver: str = "linprog",
    atarget_visual_top_k: int = 32,
) -> dict[str, Any]:
    """Compute DGST-T layer features from raw wrapper captures."""
    source_ffn = dgst_t_raw["source_ffn_states"].float()
    prediction_hidden = dgst_t_raw["prediction_hidden_states"].float()
    support_states = dgst_t_raw["support_h_mid_states"].float()
    support_attentions = dgst_t_raw["support_attentions"].float()
    semantic_probs = dgst_t_raw["semantic_probs"].float()
    prompt_last = dgst_t_raw["prompt_last_hidden_states"].float()
    prompt_mean = dgst_t_raw["prompt_mean_hidden_states"].float()
    prompt_confidence = dgst_t_raw["prompt_logit_lens_top3_confidence"].float()
    target_embedding = dgst_t_raw["target_embedding"].float()
    support_positions = [int(pos) for pos in dgst_t_raw["support_positions"]]
    visual_start = int(dgst_t_raw["visual_start"])
    visual_end = int(dgst_t_raw["visual_end"])

    if source_ffn.ndim != 2 or support_states.ndim != 3:
        raise ValueError("DGST-T raw tensors have invalid shapes.")
    if source_ffn.shape[0] != support_states.shape[0]:
        raise ValueError("DGST-T layer count mismatch between source and support states.")

    layer_stats = []
    risk_per_layer = []
    prompt_last_cosine_per_layer = []
    prompt_mean_cosine_per_layer = []
    context_confidence_per_layer = []
    atarget_visual_cosine_per_layer = []

    for layer_idx in range(int(source_ffn.shape[0])):
        layer_support_states = support_states[layer_idx]
        source_dist = _source_distribution(
            source_update=source_ffn[layer_idx],
            support_states=layer_support_states,
            tau=tau,
        )
        attention_dist = _renormalize(support_attentions[layer_idx])
        target_dist = _renormalize(attention_dist * semantic_probs[layer_idx])

        support = _topk_union_indices(source_dist, target_dist, transport_top_k)
        local_source = _renormalize(source_dist.index_select(0, support))
        local_target = _renormalize(target_dist.index_select(0, support))
        local_states = layer_support_states.index_select(0, support)
        local_semantic = semantic_probs[layer_idx].index_select(0, support)

        distance = _cosine_distance_matrix(local_states)
        source_penalty = torch.relu(1.0 - local_semantic)
        target_penalty = torch.relu(1.0 - local_semantic)
        cost = _build_cost_matrix(
            distance,
            source_penalty,
            target_penalty,
            cost_mode=cost_mode,
            lambda_d=lambda_d,
            lambda_s=lambda_s,
            lambda_t=lambda_t,
            lambda_int=lambda_int,
        )
        transport_risk, _transport_plan = _wasserstein_1_exact(
            local_source,
            local_target,
            cost,
            solver=ot_solver,
        )

        prompt_last_cosine = float(
            F.cosine_similarity(
                prediction_hidden[layer_idx].unsqueeze(0),
                prompt_last[layer_idx].unsqueeze(0),
                dim=-1,
            ).item()
        )
        prompt_mean_cosine = float(
            F.cosine_similarity(
                prediction_hidden[layer_idx].unsqueeze(0),
                prompt_mean[layer_idx].unsqueeze(0),
                dim=-1,
            ).item()
        )
        atarget_visual_cosine = _atarget_topk_visual_cosine(
            target_embedding=target_embedding,
            support_states=layer_support_states,
            target_dist=target_dist,
            support_positions=support_positions,
            visual_start=visual_start,
            visual_end=visual_end,
            top_k=atarget_visual_top_k,
        )
        context_confidence = float(prompt_confidence[layer_idx].item() * atarget_visual_cosine)

        risk_per_layer.append(float(transport_risk))
        prompt_last_cosine_per_layer.append(prompt_last_cosine)
        prompt_mean_cosine_per_layer.append(prompt_mean_cosine)
        atarget_visual_cosine_per_layer.append(float(atarget_visual_cosine))
        context_confidence_per_layer.append(context_confidence)
        layer_stats.append(
            {
                "layer": int(layer_idx + 1),
                "transport_risk": float(transport_risk),
                "prompt_last_cosine": prompt_last_cosine,
                "prompt_mean_cosine": prompt_mean_cosine,
                "prompt_logit_lens_top3_confidence": float(prompt_confidence[layer_idx].item()),
                "atarget_top32_visual_cosine": float(atarget_visual_cosine),
                "context_confidence": context_confidence,
                "support_size": int(len(support_positions)),
                "selected_support_size": int(support.numel()),
            }
        )

    risk_tensor = torch.tensor(risk_per_layer, dtype=torch.float32)
    final_score = _baseline_excess_score(
        risk_tensor,
        baseline_layers=baseline_layers,
        risk_start_layer=risk_start_layer,
        alpha=alpha,
    )

    return {
        "dgst_t_score": float(final_score),
        "dgst_t_per_layer": risk_tensor,
        "dgst_t_transport_risk_per_layer": risk_tensor,
        "dgst_t_prompt_last_cosine_per_layer": torch.tensor(prompt_last_cosine_per_layer, dtype=torch.float32),
        "dgst_t_prompt_mean_cosine_per_layer": torch.tensor(prompt_mean_cosine_per_layer, dtype=torch.float32),
        "dgst_t_atarget_visual_cosine_per_layer": torch.tensor(atarget_visual_cosine_per_layer, dtype=torch.float32),
        "dgst_t_context_confidence_per_layer": torch.tensor(context_confidence_per_layer, dtype=torch.float32),
        "dgst_t_layer_stats": layer_stats,
        "dgst_t_feature_vector": _feature_vector(
            risk_per_layer,
            prompt_last_cosine_per_layer,
            prompt_mean_cosine_per_layer,
            context_confidence_per_layer,
        ),
    }


def _feature_vector(
    risk_per_layer: Sequence[float],
    prompt_last_cosine_per_layer: Sequence[float],
    prompt_mean_cosine_per_layer: Sequence[float],
    context_confidence_per_layer: Sequence[float],
) -> list[float]:
    return [
        *[float(value) for value in risk_per_layer],
        *[float(value) for value in prompt_last_cosine_per_layer],
        *[float(value) for value in prompt_mean_cosine_per_layer],
        *[float(value) for value in context_confidence_per_layer],
    ]


def _source_distribution(
    *,
    source_update: torch.Tensor,
    support_states: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    update = source_update.float()
    states = support_states.float()
    scores = F.cosine_similarity(update.unsqueeze(0), states, dim=-1)
    return torch.softmax(scores / max(float(tau), 1e-6), dim=-1)


def _renormalize(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = values.sum()
    if total <= EPS:
        return torch.full_like(values, 1.0 / max(int(values.numel()), 1))
    return values / total


def _topk_union_indices(source: torch.Tensor, target: torch.Tensor, top_k: int) -> torch.Tensor:
    k_half = min(max(int(top_k // 2), 1), int(source.numel()))
    src_idx = torch.topk(source, k=k_half).indices
    tgt_idx = torch.topk(target, k=k_half).indices
    return torch.unique(torch.cat([src_idx, tgt_idx], dim=0), sorted=True)


def _cosine_distance_matrix(states: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(states.float(), dim=-1)
    cosine = torch.matmul(normalized, normalized.transpose(0, 1))
    return (1.0 - cosine).clamp_min(0.0)


def _build_cost_matrix(
    distance: torch.Tensor,
    source_penalty: torch.Tensor,
    target_penalty: torch.Tensor,
    *,
    cost_mode: str,
    lambda_d: float,
    lambda_s: float,
    lambda_t: float,
    lambda_int: float,
) -> torch.Tensor:
    mode = str(cost_mode).strip().lower()
    source = source_penalty.float().unsqueeze(1)
    target = target_penalty.float().unsqueeze(0)
    interaction = distance.float() * (source + target)
    if mode == "decomposed":
        return float(lambda_d) * distance + float(lambda_int) * interaction
    if mode == "direct":
        return (
            float(lambda_d) * distance
            + float(lambda_s) * source
            + float(lambda_t) * target
            + float(lambda_int) * interaction
        )
    raise ValueError("DGST-T only keeps direct/decomposed transport costs.")


def _atarget_topk_visual_cosine(
    *,
    target_embedding: torch.Tensor,
    support_states: torch.Tensor,
    target_dist: torch.Tensor,
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    top_k: int,
) -> float:
    visual_indices = [
        offset
        for offset, position in enumerate(support_positions)
        if int(visual_start) <= int(position) < int(visual_end)
    ]
    if not visual_indices:
        return 0.0
    visual_index = torch.tensor(visual_indices, dtype=torch.long)
    visual_scores = target_dist.index_select(0, visual_index)
    k = min(max(int(top_k), 1), int(visual_scores.numel()))
    selected = visual_index.index_select(0, torch.topk(visual_scores, k=k).indices)
    selected_states = support_states.index_select(0, selected).float()
    similarities = F.cosine_similarity(target_embedding.float().unsqueeze(0), selected_states, dim=-1)
    return float(similarities.mean().item())


def _baseline_excess_score(
    risk_per_layer: torch.Tensor,
    *,
    baseline_layers: int,
    risk_start_layer: int,
    alpha: float,
) -> float:
    if risk_per_layer.numel() == 0:
        return 0.0
    base_width = max(1, min(int(baseline_layers), int(risk_per_layer.numel())))
    baseline = risk_per_layer[:base_width]
    baseline_mean = float(baseline.mean().item())
    baseline_std = float(baseline.std(unbiased=False).item())
    threshold = baseline_mean + float(alpha) * baseline_std
    score = 0.0
    for layer_number, value in enumerate(risk_per_layer.tolist(), start=1):
        if layer_number >= int(risk_start_layer):
            score += max(0.0, float(value) - threshold)
    return float(score)


def _wasserstein_1_exact(
    source: torch.Tensor,
    target: torch.Tensor,
    cost: torch.Tensor,
    *,
    solver: str,
) -> tuple[float, torch.Tensor]:
    solver_name = str(solver).strip().lower()
    if solver_name not in {"linprog", "emd"}:
        raise ValueError("DGST-T ot_solver must be 'linprog' or 'emd'.")

    try:
        import numpy as np
    except Exception as exc:
        raise ImportError("DGST-T exact OT requires numpy.") from exc

    source_np = _as_probability_numpy(source)
    target_np = _as_probability_numpy(target)
    source_np, target_np = _balance_ot_marginals(source_np, target_np)
    cost_np = torch.nan_to_num(
        cost.detach().cpu().double(),
        nan=1e6,
        posinf=1e6,
        neginf=1e6,
    ).clamp_min(0.0).numpy()

    if solver_name == "emd":
        try:
            import ot
        except Exception as exc:
            raise ImportError("DGST-T exact EMD solver requires POT.") from exc
        plan_array = ot.emd(source_np, target_np, cost_np)
        plan = torch.tensor(plan_array, dtype=torch.float64)
        return float((plan * torch.tensor(cost_np, dtype=torch.float64)).sum().item()), plan

    try:
        from scipy.optimize import linprog
    except Exception as exc:
        raise ImportError("DGST-T exact OT requires scipy.") from exc

    n = int(source_np.shape[0])
    result = linprog(
        c=cost_np.reshape(-1),
        A_eq=_transport_constraint_matrix(n),
        b_eq=np.concatenate([source_np, target_np[:-1]]),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        source_np, target_np = _balance_ot_marginals(source_np, target_np, slack=1e-10)
        result = linprog(
            c=cost_np.reshape(-1),
            A_eq=_transport_constraint_matrix(n),
            b_eq=np.concatenate([source_np, target_np[:-1]]),
            bounds=(0.0, None),
            method="highs",
        )
    if not result.success:
        try:
            import ot
            plan_array = ot.emd(source_np, target_np, cost_np)
            plan = torch.tensor(plan_array, dtype=torch.float64)
            return float((plan * torch.tensor(cost_np, dtype=torch.float64)).sum().item()), plan
        except Exception as exc:
            raise RuntimeError(f"DGST-T exact OT solver failed: {result.message}") from exc
    plan = torch.tensor(result.x, dtype=torch.float64).reshape(n, n)
    return float(result.fun), plan


def _as_probability_numpy(values: torch.Tensor):
    import numpy as np

    arr = values.detach().cpu().double().numpy().astype(np.float64, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0.0] = 0.0
    total = float(arr.sum())
    if total <= 0.0:
        arr.fill(1.0 / max(int(arr.size), 1))
    else:
        arr /= total
    arr[-1] += 1.0 - float(arr.sum())
    if arr[-1] < 0.0:
        arr[arr < 0.0] = 0.0
        arr /= max(float(arr.sum()), EPS)
        arr[-1] += 1.0 - float(arr.sum())
    return arr


def _balance_ot_marginals(source, target, *, slack: float = 1e-12):
    """Make the dropped-last-column OT constraints numerically feasible."""
    import numpy as np

    source = np.array(source, dtype=np.float64, copy=True)
    target = np.array(target, dtype=np.float64, copy=True)
    source[source < 0.0] = 0.0
    target[target < 0.0] = 0.0
    source /= max(float(source.sum()), EPS)
    target /= max(float(target.sum()), EPS)
    source[-1] += 1.0 - float(source.sum())

    if target.size == 1:
        target[0] = float(source.sum())
        return source, target

    prefix_sum = float(target[:-1].sum())
    source_sum = float(source.sum())
    max_prefix = max(0.0, source_sum - float(slack))
    if prefix_sum > max_prefix:
        target[:-1] *= max_prefix / max(prefix_sum, EPS)
    target[-1] = max(0.0, source_sum - float(target[:-1].sum()))
    target /= max(float(target.sum()), EPS)

    prefix_sum = float(target[:-1].sum())
    if prefix_sum > source_sum:
        target[:-1] *= source_sum / max(prefix_sum, EPS)
        target[-1] = 0.0
    return source, target


@lru_cache(maxsize=32)
def _transport_constraint_matrix(size: int):
    try:
        import numpy as np
        from scipy import sparse
    except Exception as exc:
        raise ImportError("DGST-T exact OT requires numpy and scipy.") from exc

    n = int(size)
    row_count = (2 * n) - 1
    row_indices: list[int] = []
    col_indices: list[int] = []

    for row_idx in range(n):
        row_indices.extend([row_idx] * n)
        col_indices.extend([row_idx * n + col_idx for col_idx in range(n)])
    for col_idx in range(n - 1):
        row_indices.extend([n + col_idx] * n)
        col_indices.extend([row_idx * n + col_idx for row_idx in range(n)])

    data = np.ones(len(row_indices), dtype=np.float64)
    return sparse.csr_matrix((data, (row_indices, col_indices)), shape=(row_count, n * n))
