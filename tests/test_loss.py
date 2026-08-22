import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from w2sg.loss import alpha_schedule, w2s_listwise_loss


def test_alpha_zero_before_ramp_start():
    assert alpha_schedule(step=0, total_steps=1000, ceiling=0.4, ramp_start_frac=0.3, ramp_end_frac=0.7) == 0.0
    assert alpha_schedule(step=299, total_steps=1000, ceiling=0.4, ramp_start_frac=0.3, ramp_end_frac=0.7) == 0.0


def test_alpha_reaches_ceiling_and_stays():
    assert alpha_schedule(step=700, total_steps=1000, ceiling=0.4, ramp_start_frac=0.3, ramp_end_frac=0.7) == 0.4
    assert alpha_schedule(step=999, total_steps=1000, ceiling=0.4, ramp_start_frac=0.3, ramp_end_frac=0.7) == 0.4


def test_alpha_linear_midramp():
    mid = alpha_schedule(step=500, total_steps=1000, ceiling=0.4, ramp_start_frac=0.3, ramp_end_frac=0.7)
    assert math.isclose(mid, 0.2, abs_tol=1e-6)  # halfway through the 0.3->0.7 window


def test_loss_alpha_zero_ignores_self_term():
    torch.manual_seed(0)
    logits = torch.randn(4, 6)
    target = torch.softmax(torch.randn(4, 6), dim=-1)
    conf = torch.ones(4, 6)

    loss_alpha0 = w2s_listwise_loss(logits, target, conf, alpha=0.0)
    logp = torch.log_softmax(logits, dim=-1)
    expected = -(conf * target * logp).sum(-1).mean()
    assert torch.isclose(loss_alpha0, expected, atol=1e-6)


def test_loss_alpha_one_ignores_weak_target():
    torch.manual_seed(1)
    logits = torch.randn(4, 6)
    conf = torch.ones(4, 6)
    zero_target = torch.zeros(4, 6)
    real_target = torch.softmax(torch.randn(4, 6), dim=-1)

    loss_zero_target = w2s_listwise_loss(logits, zero_target, conf, alpha=1.0)
    loss_real_target = w2s_listwise_loss(logits, real_target, conf, alpha=1.0)
    assert torch.isclose(loss_zero_target, loss_real_target, atol=1e-6)


def test_self_term_is_detached():
    logits = torch.randn(3, 5, requires_grad=True)
    target = torch.softmax(torch.randn(3, 5), dim=-1)
    conf = torch.ones(3, 5)

    loss = w2s_listwise_loss(logits, target, conf, alpha=0.5, tau=0.5)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_gated_candidates_contribute_zero_weak_gradient():
    torch.manual_seed(2)
    logits = torch.randn(2, 4)
    target = torch.softmax(torch.randn(2, 4), dim=-1)

    conf_full = torch.ones(2, 4)
    conf_one_gated = conf_full.clone()
    conf_one_gated[0, 0] = 0.0

    logp = torch.log_softmax(logits, dim=-1)
    weak_full = -(conf_full * target * logp).sum(-1)
    weak_gated = -(conf_one_gated * target * logp).sum(-1)
    assert not torch.isclose(weak_full[0], weak_gated[0])
    assert torch.isclose(weak_full[1], weak_gated[1])


def test_padding_mask_does_not_produce_nan():
    logits = torch.randn(2, 5)
    target = torch.zeros(2, 5)
    target[:, :3] = torch.softmax(torch.randn(2, 3), dim=-1)
    conf = torch.zeros(2, 5)
    conf[:, :3] = 1.0
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, :3] = True

    loss = w2s_listwise_loss(logits, target, conf, alpha=0.5, mask=mask)
    assert torch.isfinite(loss)


def test_nan_logits_raise_not_silently_propagate():
    logits = torch.tensor([[float("nan"), 0.0, 0.0]])
    target = torch.softmax(torch.randn(1, 3), dim=-1)
    conf = torch.ones(1, 3)
    with pytest.raises(FloatingPointError):
        w2s_listwise_loss(logits, target, conf, alpha=0.3)
