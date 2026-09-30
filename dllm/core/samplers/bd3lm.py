"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import copy
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


def _prepare_for_sampling(
    x: torch.Tensor,
    block_size: int,
    pad_token_id: int,
    q_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a block-wise bidirectional attention mask and position_ids
    over the entire sequence (prompt + generated).

    Padding tokens (pad_token_id) are excluded from attention: they are neither
    valid queries nor valid keys.

    Block boundaries are defined in *physical* coordinates (shared across batch):
      - block_id[pos] = pos // block_size   for column index pos = 0..T-1

    For position_ids (used by RoPE), we still use per-sample logical positions:
      - valid[b, t] = (x[b, t] != pad_token_id)
      - pos_raw[b, t]  = count of valid tokens up to and including t (1-based)
      - logical_pos[b, t] = pos_raw[b, t] - 1, for valid positions

    Only the query rows `q_start:` of the mask are built (keys are always all T).

    Returns:
        attn_mask: [B, 1, T - q_start, T] bool
        position_ids: [B, T] long, logical positions (padding set to 0)
    """
    B, T = x.shape
    device = x.device

    # Per-sample valid mask
    valid = x != pad_token_id  # [B, T]

    # Per-sample logical positions for RoPE (skip padding)
    pos_raw = torch.cumsum(valid.to(torch.long), dim=-1)  # [B, T], 1-based
    logical_pos = pos_raw - 1  # [B, T], 0-based

    # Position ids: logical positions for valid tokens, 0 for padding
    position_ids = torch.where(
        valid,
        logical_pos,
        torch.zeros_like(logical_pos),
    ).to(
        device=device, dtype=torch.long
    )  # [B, T]

    # Block ids for attention: defined in physical coordinates
    pos = torch.arange(T, device=device)  # [T]
    block_ids = torch.div(pos, block_size, rounding_mode="floor")  # [T]
    block_ids = block_ids.view(1, T).expand(B, -1)  # [B, T]

    # Mark padding positions as "no block"
    block_ids = torch.where(
        valid,
        block_ids,
        torch.full_like(block_ids, -1),
    )

    # Build [B, 1, T - q_start, T] mask
    bid_q = block_ids[:, q_start:].view(B, 1, T - q_start, 1)  # query
    bid_k = block_ids.view(B, 1, 1, T)  # key

    valid_q = bid_q >= 0
    valid_k = bid_k >= 0

    base_mask = bid_k <= bid_q
    attn_mask = base_mask & valid_q & valid_k  # [B, 1, T - q_start, T]

    return attn_mask, position_ids


def _fork_cache(cache):
    """Cache to give a step that must not change `cache`.

    HF-style caches are updated in place by the forward pass, so those that support
    `crop` are passed as is and rolled back afterwards (`_rollback_cache`). Immutable
    (tuple) caches need nothing; anything else is deep-copied.
    """
    if hasattr(cache, "crop") or isinstance(cache, tuple):
        return cache
    return copy.deepcopy(cache)


def _rollback_cache(cache, length: int) -> None:
    if hasattr(cache, "crop"):
        cache.crop(length)


def _diffusion_step_block(
    logits: torch.Tensor,  # [B, L, V]
    x_block: torch.Tensor,  # [B, L]
    mask_block: torch.Tensor,  # [B, L] bool
    num_transfer_step: torch.Tensor,  # [B]
    k_max: int,  # max of num_transfer_step, as a Python int
    temperature: float,
    remasking: str,
) -> torch.Tensor:
    """
    One diffusion step over a block slice [B, L].
    """
    # Gumbel-max sampling
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

    # Confidence
    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=logits.device)
    else:
        raise NotImplementedError(remasking)

    # Only masked positions can change
    x0 = torch.where(mask_block, x0, x_block)
    confidence = torch.where(mask_block, x0_p, -torch.inf)

    # Pick positions to commit
    transfer = topk_transfer_mask(confidence, num_transfer_step, k_max)
    return torch.where(transfer, x0, x_block)


@dataclass
class BD3LMSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_size: int = 32
    steps: int = 128
    steps_per_block: int | None = None
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    right_shift_logits: bool = False


@dataclass
class BD3LMSampler(BaseSampler):

    def _extend_cache(self, tokens, attn, pos, past):
        """Run `tokens` through the model on top of `past` and keep the updated cache.
        Returns (cache, logits of the last token) — the latter feeds `right_shift_logits`."""
        out = self.model(
            tokens,
            attention_mask=attn,
            position_ids=pos,
            past_key_values=past,
            use_cache=True,
        )
        return out.past_key_values, out.logits[:, -1:, :]

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: BD3LMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text using block diffusion language modeling.

        Generates text block-by-block with an attention pattern, where each
        block undergoes multiple diffusion steps before moving to the next block.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """

        config = (config or BD3LMSamplerConfig()).override(**kwargs)
        max_new_tokens = config.max_new_tokens
        max_length = config.max_length
        steps_per_block = config.steps_per_block

        assert config.block_size >= 1
        assert config.steps >= 1

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        pad_id = self.tokenizer.pad_token_id  # used as padding here
        eos_id = self.tokenizer.eos_token_id

        # ---- normalize inputs to tensors ----
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

        # Decide how many new tokens to generate
        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        max_prompt_len = max(prompt_lens)

        # ==========================================================
        # 1) Initialize with prompt only (left padded with pad_id)
        #    pad prefix length to a multiple of block_size
        # ==========================================================
        device = self.model.device
        padded_prompt_len = (
            (max_prompt_len + config.block_size - 1) // config.block_size
        ) * config.block_size

        prompt_lens_t = torch.tensor(prompt_lens, device=device)[:, None]  # [B, 1]
        cols = torch.arange(padded_prompt_len, device=device)[None, :]  # [1, P]
        src = cols - (padded_prompt_len - prompt_lens_t)  # < 0 on the left padding
        prompts = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(
            device=device, dtype=torch.long
        )
        x = torch.where(src >= 0, prompts.gather(1, src.clamp(min=0)), pad_id)

        # Tokens considered "given" for unconditional branch in CFG.
        unmasked_index = (x != mask_id) & (x != pad_id)
        if config.cfg_keep_tokens:
            keep_mask = torch.isin(
                x, torch.as_tensor(config.cfg_keep_tokens, device=device)
            )
            unmasked_index = unmasked_index & (~keep_mask)

        # track done per sequence (EOS)
        done = torch.zeros((B,), dtype=torch.bool, device=device)

        # ---- block scheduling ----
        num_blocks = math.ceil(max_new_tokens / config.block_size)
        if steps_per_block is None:
            steps_per_block = math.ceil(config.steps / num_blocks)
        histories = [x.clone()] if config.return_dict else None

        # ==========================================================
        # 2) Encode the prompt once. Blocks are causal across blocks, so a finished
        #    block's keys/values never change: later blocks only extend this cache.
        # ==========================================================
        prompt_attn, prompt_pos = _prepare_for_sampling(
            x=x, block_size=config.block_size, pad_token_id=pad_id
        )  # [B,1,P,P], [B,P]
        cond_past, cond_last = self._extend_cache(x, prompt_attn, prompt_pos, None)
        if config.cfg_scale > 0.0:
            un_x = torch.where(unmasked_index, mask_id, x)
            uncond_past, uncond_last = self._extend_cache(
                un_x, prompt_attn, prompt_pos, None
            )

        generated = 0  # number of generated tokens so far

        # ==========================================================
        # 3) Block-by-block generation loop
        # ==========================================================
        for b_idx in range(num_blocks):
            T_prefix = x.shape[1]  # current total length before appending this block

            # With padded_prompt_len aligned, we always append whole blocks (except possibly final)
            cur_block_len = min(config.block_size, max_new_tokens - generated)
            if cur_block_len <= 0:
                break

            new_block = torch.full(
                (B, cur_block_len), mask_id, dtype=torch.long, device=device
            )
            x = torch.cat([x, new_block], dim=1)  # [B, T_prefix + cur_block_len]
            T_total = x.shape[1]

            block_mask_index = x[:, -cur_block_len:] == mask_id  # [B, cur_block_len]

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=config.stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)
            k_max_per_step = num_transfer_tokens.max(dim=0).values.tolist()

            # Attention rows / positions of the current block only
            attn_block, full_position_ids = _prepare_for_sampling(
                x=x,
                block_size=config.block_size,
                pad_token_id=pad_id,
                q_start=T_prefix,
            )  # [B,1,L_q,T_total], [B,T_total]
            pos_block = full_position_ids[:, T_prefix:T_total]  # [B,L_q]

            # Global AR-style right shift across blocks: logits of the last prefix token
            if config.right_shift_logits:
                if config.cfg_scale > 0.0:
                    prefix_last_logits = uncond_last + (config.cfg_scale + 1.0) * (
                        cond_last - uncond_last
                    )  # [B, 1, V]
                else:
                    prefix_last_logits = cond_last  # [B, 1, V]

            # ======================================================
            # 4) Inner diffusion loop within the current block
            # ======================================================
            for i_step in range(effective_steps):
                x_block = x[:, T_prefix:T_total]  # [B, cur_block_len]
                mask_block = x_block == mask_id

                # ---- Conditional logits for current block ----
                cond_logits_block = self.model(
                    x_block,
                    attention_mask=attn_block,
                    position_ids=pos_block,
                    past_key_values=_fork_cache(cond_past),
                    use_cache=False,
                ).logits  # [B, cur_block_len, V]
                _rollback_cache(cond_past, T_prefix)

                logits_block = cond_logits_block

                # ---- Optional CFG ----
                if config.cfg_scale > 0.0:
                    un_logits_block = self.model(
                        x_block,
                        attention_mask=attn_block,
                        position_ids=pos_block,
                        past_key_values=_fork_cache(uncond_past),
                        use_cache=False,
                    ).logits  # [B, cur_block_len, V]
                    _rollback_cache(uncond_past, T_prefix)

                    logits_block = un_logits_block + (config.cfg_scale + 1.0) * (
                        cond_logits_block - un_logits_block
                    )

                if config.right_shift_logits:
                    shifted = torch.empty_like(logits_block)
                    shifted[:, 0:1, :] = prefix_last_logits
                    shifted[:, 1:, :] = logits_block[:, :-1, :]
                    logits_block = shifted

                # ---- One diffusion step over this block ----
                x_block_updated = _diffusion_step_block(
                    logits=logits_block,
                    x_block=x_block,
                    mask_block=mask_block,
                    num_transfer_step=num_transfer_tokens[:, i_step],
                    k_max=k_max_per_step[i_step],
                    temperature=config.temperature,
                    remasking=config.remasking,
                )

                # Write back
                x[:, T_prefix:T_total] = x_block_updated

                if histories is not None:
                    histories.append(x.clone())

            # per-sequence EOS stopping (after finishing denoising the block)
            if eos_id is not None:
                eos_in_block = (x[:, T_prefix:T_total] == eos_id).any(dim=1)
                done = done | eos_in_block

            generated += cur_block_len
            if b_idx + 1 == num_blocks or done.all():
                break

            # Extend the caches with the finished block (only its tokens go through the model)
            x_block = x[:, T_prefix:T_total]
            cond_past, cond_last = self._extend_cache(
                x_block, attn_block, pos_block, cond_past
            )
            if config.cfg_scale > 0.0:
                uncond_past, uncond_last = self._extend_cache(
                    x_block, attn_block, pos_block, uncond_past
                )

        # ==========================================================
        # 4) Output
        # ==========================================================
        if not config.return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: BaseSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:
        raise NotImplementedError