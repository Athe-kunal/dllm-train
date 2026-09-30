"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import (
    add_gumbel_noise,
    get_num_transfer_tokens,
    topk_transfer_mask,
)


def _token_ids(tokens: list[int] | None, device) -> torch.Tensor | None:
    return torch.tensor(tokens, device=device) if tokens else None


def _window_logits(
    logits: torch.Tensor, c0: int, c1: int, right_shift: bool
) -> torch.Tensor:
    """Columns [c0, c1) of `logits` [B, T, V]; with `right_shift`, column i holds the
    logits of column i-1 (column 0 keeps its own)."""
    if not right_shift:
        return logits[:, c0:c1]
    src = torch.arange(c0 - 1, c1 - 1, device=logits.device).clamp(min=0)
    return logits[:, src]


@dataclass
class MDLMSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False


@dataclass
class MDLMSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = MDLMSamplerConfig()

        config = (config or MDLMSamplerConfig()).override(**kwargs)
        max_new_tokens = config.max_new_tokens
        max_length = config.max_length

        assert 1 <= config.block_size
        assert 1 <= config.steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if config.right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length
        device = self.model.device
        cols = torch.arange(T, device=device)[None, :]  # [1, T]
        prompt_lens_t = torch.tensor(prompt_lens, device=device)[:, None]  # [B, 1]

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=device)
        x[:, : max(prompt_lens)] = pad_sequence(
            inputs, batch_first=True, padding_value=eos_id
        )
        x.masked_fill_(
            (cols >= prompt_lens_t) & (cols < prompt_lens_t + max_new_tokens), mask_id
        )
        attention_mask = (cols < prompt_lens_t + max_new_tokens).long()

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (config.cfg_keep_tokens is None or len(config.cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(config.cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / config.block_size)
        steps_per_block = math.ceil(config.steps / num_blocks)  # per-block step budget
        histories = [x.clone()] if config.return_dict else None

        suppress_ids = _token_ids(config.suppress_tokens, x.device)
        begin_suppress_ids = _token_ids(config.begin_suppress_tokens, x.device)

        for b in range(num_blocks):
            # [B, T] window of the current block, aligned to each prompt's tail
            lo = prompt_lens_t + b * config.block_size
            hi = torch.clamp(lo + config.block_size, max=prompt_lens_t + max_new_tokens)
            window = (cols >= lo) & (cols < hi)
            # Column range covering every row's window; post-processing only touches it
            c0 = min(prompt_lens) + b * config.block_size
            c1 = max(
                min(pl + (b + 1) * config.block_size, pl + max_new_tokens)
                for pl in prompt_lens
            )
            window_w = window[:, c0:c1]

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=(x == mask_id) & window,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=config.stochastic_transfer,
            )

            # Some steps may be skipped if there are no transfers
            effective_steps = num_transfer_tokens.size(1)
            k_max_per_step = num_transfer_tokens.max(dim=0).values.tolist()

            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                # Optional CFG: second forward where original prompt tokens are masked out
                if config.cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = _window_logits(logits, c0, c1, config.right_shift_logits)
                    un_logits = _window_logits(un_logits, c0, c1, config.right_shift_logits)
                    logits = un_logits + (config.cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = _window_logits(
                        self.model(x, attention_mask=attention_mask).logits,
                        c0,
                        c1,
                        config.right_shift_logits,
                    )  # [B, W, V]

                if suppress_ids is not None:
                    logits[:, :, suppress_ids] = -torch.inf

                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, W]

                if begin_suppress_ids is not None:
                    logits[:, :, begin_suppress_ids] = -torch.inf

                # Per-position confidence used to pick which masks to commit this step
                if config.remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, W] confidence of predicted token
                elif config.remasking == "random":
                    x0_p = torch.rand((B, T), device=x.device)[:, c0:c1]
                else:
                    raise NotImplementedError(config.remasking)

                x_w = x[:, c0:c1]
                mask_w = x_w == mask_id
                x0 = torch.where(mask_w, x0, x_w)
                confidence = torch.where(mask_w & window_w, x0_p, -torch.inf)

                # Pick exactly `num_transfer_tokens[j, i]` highest-confidence positions per sample
                transfer_index = topk_transfer_mask(
                    confidence, num_transfer_tokens[:, i], k_max_per_step[i]
                )

                # Commit chosen predictions into the canvas
                x[:, c0:c1] = torch.where(transfer_index, x0, x_w)
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not config.return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        config = (config or MDLMSamplerConfig()).override(**kwargs)
        block_size = config.block_size

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if config.right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Default to a single block spanning the whole sequence
        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= config.steps

        device = self.model.device
        cols = torch.arange(T, device=device)[None, :]  # [1, T]
        seq_lens_t = torch.tensor(seq_lens, device=device)[:, None]  # [B, 1]

        x = pad_sequence(inputs, batch_first=True, padding_value=eos_id).to(
            device=device, dtype=torch.long
        )
        attention_mask = (cols < seq_lens_t).long()

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (config.cfg_keep_tokens is None or len(config.cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(config.cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(config.steps / num_blocks)
        histories = [x.clone()] if config.return_dict else None

        suppress_ids = _token_ids(config.suppress_tokens, x.device)
        begin_suppress_ids = _token_ids(config.begin_suppress_tokens, x.device)

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # [B, T] window of the current block, limited by each sample's true length
            window = (cols >= start) & (cols < torch.clamp(seq_lens_t, max=stop))
            window_w = window[:, start:stop]

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=(x == mask_id) & window,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=config.stochastic_transfer,
            )

            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)
            k_max_per_step = num_transfer_tokens.max(dim=0).values.tolist()

            for s in range(effective_steps):
                # ----- Forward pass (+ optional CFG) -----
                if config.cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = _window_logits(logits, start, stop, config.right_shift_logits)
                    un_logits = _window_logits(
                        un_logits, start, stop, config.right_shift_logits
                    )
                    logits = un_logits + (config.cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = _window_logits(
                        self.model(x, attention_mask=attention_mask).logits,
                        start,
                        stop,
                        config.right_shift_logits,
                    )  # [B, W, V]

                if suppress_ids is not None:
                    logits[:, :, suppress_ids] = -torch.inf

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, W]

                if begin_suppress_ids is not None:
                    logits[:, :, begin_suppress_ids] = -torch.inf

                # Confidence used for choosing which masks to commit this step
                if config.remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, W]
                elif config.remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)[:, start:stop]
                else:
                    raise NotImplementedError(config.remasking)

                x_w = x[:, start:stop]
                mask_w = x_w == mask_id
                x0 = torch.where(mask_w, x0, x_w)
                confidence = torch.where(mask_w & window_w, x0_p, -torch.inf)

                # Pick exactly num_transfer_tokens[j, s] positions per sample
                transfer_index = topk_transfer_mask(
                    confidence, num_transfer_tokens[:, s], k_max_per_step[s]
                )

                # Commit selected predictions into the canvas
                x[:, start:stop] = torch.where(transfer_index, x0, x_w)
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not config.return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)