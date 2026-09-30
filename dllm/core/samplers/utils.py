import torch

from dllm.core.schedulers import BaseAlphaScheduler


def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: BaseAlphaScheduler,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Compute the number of tokens to unmask at each diffusion step.

    For each sample, determines how many masked tokens should be revealed
    per step based on the reverse diffusion schedule.

    Args:
        mask_index: Boolean tensor [B, L] indicating masked positions.
        steps: Number of diffusion steps.
        scheduler: Alpha scheduler defining the masking schedule.
        stochastic: If True, sample from a binomial distribution (probabilistic);
            if False, use deterministic rounding of the expected number of tokens.

    Returns:
        Integer tensor [B, steps] with number of tokens to unmask per step.
    """
    device = mask_index.device
    # The whole computation is a few tiny [B]-sized ops, so run it on the CPU: one
    # device->host copy here replaces a kernel launch per op per step (and the later
    # `.item()` sync). The result is moved back to `device` at the end.
    mask_num = mask_index.sum(dim=1).to("cpu", torch.float64)  # [B]
    B = mask_num.size(0)

    # The reveal probability of every step depends only on (j, steps, scheduler), not on
    # how many tokens remain, so compute all of them with ONE scheduler call on [steps]
    # vectors instead of `steps` scalar calls (each of which builds tensors + validates).
    j = torch.arange(steps, dtype=torch.float64)
    s = (steps - 1 - j) / steps
    t = (steps - j) / steps
    reverse_transfer_prob = (1 - scheduler.reverse_mask_prob(s=s, t=t)).to(
        torch.float64
    )  # [steps]

    # This loop is inherently sequential: step j reveals `round(remaining * p_j)`, and
    # `remaining` depends on the rounding done in every earlier step. It is vectorised
    # over the batch, and now only touches [B]-sized CPU tensors.
    num_transfer_tokens = torch.zeros(B, steps, dtype=torch.int64)
    for j, p in enumerate(reverse_transfer_prob.tolist()):
        if not stochastic:
            n_step = torch.round(mask_num * p)
        else:
            n_step = torch.distributions.Binomial(
                mask_num, torch.as_tensor(p, dtype=torch.float64)
            ).sample()
        n_step = torch.minimum(n_step, mask_num)
        num_transfer_tokens[:, j] = n_step.to(torch.int64)
        mask_num = mask_num - n_step

    # Note: because llada is not conditioned on time, this allows us to skip steps with no unmasking (i.e. transfer).
    # Left-pack the non-zero entries of each row (stable), then trim to the widest row.
    nonzero = num_transfer_tokens > 0
    order = torch.argsort((~nonzero).to(torch.uint8), dim=1, stable=True)
    packed = torch.gather(num_transfer_tokens, 1, order)
    max_len = int(nonzero.sum(dim=1).max())
    return packed[:, :max_len].to(device)


def topk_transfer_mask(
    confidence: torch.Tensor, k: torch.Tensor, k_max: int
) -> torch.Tensor:
    """
    Boolean mask [B, L] selecting the top-`k[b]` entries of each row of `confidence`.

    Uses a single batched topk with `k_max` and masks out the surplus, instead of a
    per-row topk. Entries whose confidence is -inf are never selected.

    Args:
        confidence: Float tensor [B, L]; -inf marks ineligible positions.
        k: Integer tensor [B] with the per-row number of positions to select.
        k_max: Python int upper bound on `k` (avoids a device sync here).
    """
    transfer = torch.zeros_like(confidence, dtype=torch.bool)
    if k_max <= 0:
        return transfer
    vals, idx = torch.topk(confidence, k=min(k_max, confidence.size(1)), dim=1)
    keep = torch.arange(vals.size(1), device=k.device)[None, :] < k[:, None]
    keep &= vals > -float("inf")
    return transfer.scatter_(1, idx, keep)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    # log-domain form of `exp(logits) / (-log(noise)) ** temperature`: identical argmax,
    # but no exp() over [B, L, V] and no overflow to inf.
    return logits - temperature * torch.log(-torch.log(noise))