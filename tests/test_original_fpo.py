from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.fpo import original_fpo_ratio, ppo_objective
from algorithms.fpo_plus_plus import aspo_objective, fpo_ratio


def test_original_ratio_averages_before_exp() -> None:
    old = torch.tensor([[0.0, 2.0]])
    new = torch.zeros_like(old)
    ratio = original_fpo_ratio(old, new)
    assert ratio.shape == (1, 1)
    assert torch.allclose(ratio, torch.tensor([[torch.exp(torch.tensor(1.0))]]))


def test_original_and_fpo_plus_plus_ratios_are_structurally_different() -> None:
    old = torch.tensor([[0.0, 2.0]])
    new = torch.zeros_like(old)
    original = original_fpo_ratio(old, new)
    per_mc = fpo_ratio(old, new, delta_clip=0.0)
    assert per_mc.shape == (1, 2)
    assert float(per_mc.mean().item()) > float(original.item())


def test_original_uses_ppo_for_negative_advantage_not_spo() -> None:
    ratio = torch.tensor([[1.5]])
    advantage = torch.tensor([[-1.0]])
    original = ppo_objective(ratio, advantage, clip=0.2)
    plus_plus = aspo_objective(ratio, advantage, clip=0.2)
    expected_ppo = torch.minimum(
        ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
    )
    assert torch.allclose(original, expected_ppo)
    assert not torch.allclose(original, plus_plus)


if __name__ == "__main__":
    tests = (
        test_original_ratio_averages_before_exp,
        test_original_and_fpo_plus_plus_ratios_are_structurally_different,
        test_original_uses_ppo_for_negative_advantage_not_spo,
    )
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"All {len(tests)} original FPO tests passed.")
