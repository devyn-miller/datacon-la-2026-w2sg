"""Stage 5: the loss function."""
import torch
import torch.nn.functional as F

_MASK_FILL = -1e4


def alpha_schedule(step: int, total_steps: int, ceiling: float, ramp_start_frac: float, ramp_end_frac: float) -> float:
    """Linear ramp from 0 to ceiling after ramp_start_frac."""
    if total_steps <= 0:
        return 0.0
    frac = step / total_steps
    if frac <= ramp_start_frac:
        return 0.0
    if frac >= ramp_end_frac:
        return ceiling
    progress = (frac - ramp_start_frac) / (ramp_end_frac - ramp_start_frac)
    return ceiling * progress


def w2s_listwise_loss(
    logits: torch.Tensor,
    weak_target: torch.Tensor,
    conf: torch.Tensor,
    alpha: float,
    tau: float = 0.5,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """(1-alpha)*reliability_weighted_CE + alpha*CE(sharpen(logits).detach(), logits)"""
    if mask is not None:
        logits = logits.masked_fill(~mask, _MASK_FILL)

    logp = F.log_softmax(logits, dim=-1)
    weak_term = -(conf * weak_target * logp).sum(-1)

    with torch.no_grad():
        own = F.softmax(logits / tau, dim=-1)
    self_term = -(own * logp).sum(-1)

    per_example = (1 - alpha) * weak_term + alpha * self_term
    loss = per_example.mean()

    if torch.isnan(loss):
        raise FloatingPointError("w2s_listwise_loss produced NaN")
    return loss
