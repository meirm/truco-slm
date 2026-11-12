"""
System 1 Causal Small Language Model (SLM) for Argentine Truco ("JEV-Truco").

Architecture:
- Modern decoder-only Transformer (RMSNorm, SwiGLU feed-forward, causal self-attention)
- Zero-thinking (System 1): reads the serialized observation tokens ending at `<ACT>`
  and in a single forward pass outputs:
  1. Action logits (`[B, NUM_ACTIONS]`) with strict legal-action masking
  2. Expected payoff value (`[B]` in points)
- Supports training via:
  - Outcome-Sampling MCCFR distillation (KL-divergence to Nash equilibrium policy)
  - PPO / Actor-Critic self-play fine-tuning
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from truco.engine import Action, NUM_ACTIONS, TrucoHandState
from truco.tokenizer import TrucoTokenizer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class System1TrucoSLM(nn.Module):
    """
    Fast System-1 Small Language Model for Truco.
    Reads observation tokens and emits masked action logits + state value at the `<ACT>` token.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 384,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)

        # Single-token System 1 Action Head (maps <ACT> hidden state -> 12 action logits)
        self.policy_head = nn.Linear(d_model, NUM_ACTIONS, bias=False)
        # Single-token Value Head (maps <ACT> hidden state -> expected net points in hand)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):  # type: ignore[override]
        """
        Seamlessly upgrades older checkpoints if new Bayesian Belief tokens were appended
        to `TrucoTokenizer` (copies existing 147 token embeddings and keeps initialized rows for new tokens).
        """
        if "tok_emb.weight" in state_dict:
            old_w = state_dict["tok_emb.weight"]
            new_w = self.tok_emb.weight.data
            if old_w.shape[0] < new_w.shape[0] and old_w.shape[1] == new_w.shape[1]:
                merged = new_w.clone()
                merged[: old_w.shape[0]] = old_w.to(merged.device)
                state_dict = dict(state_dict)
                state_dict["tok_emb.weight"] = merged
        return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        input_ids: torch.Tensor,
        act_positions: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) token IDs
            act_positions: (B,) index of the `<ACT>` token in each sequence
            legal_mask: (B, NUM_ACTIONS) boolean tensor where True = legal action

        Returns:
            logits: (B, NUM_ACTIONS) with illegal actions masked to -1e9
            values: (B,) predicted expected point differential
        """
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)

        # Gather hidden state at the `<ACT>` token position for each batch item
        batch_indices = torch.arange(B, device=device)
        act_hidden = x[batch_indices, act_positions]  # (B, d_model)

        logits = self.policy_head(act_hidden)  # (B, NUM_ACTIONS)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)

        values = self.value_head(act_hidden).squeeze(-1)  # (B,)
        return logits, values

    @torch.no_grad()
    def act(
        self,
        state: TrucoHandState,
        tokenizer: TrucoTokenizer,
        device: torch.device,
        temperature: float = 0.0,
    ) -> Tuple[Action, Dict[Action, float], float]:
        """
        Zero-latency System 1 inference on a single TrucoHandState.
        Returns (chosen_action, action_probabilities_dict, expected_value).
        """
        self.eval()
        token_ids = tokenizer.encode_observation(state)
        if len(token_ids) > self.max_seq_len:
            token_ids = [token_ids[0]] + token_ids[-(self.max_seq_len - 1):]

        inp = torch.tensor([token_ids], dtype=torch.long, device=device)
        act_pos = torch.tensor([len(token_ids) - 1], dtype=torch.long, device=device)

        legal = state.legal_actions()
        mask = torch.zeros((1, NUM_ACTIONS), dtype=torch.bool, device=device)
        for a in legal:
            mask[0, int(a)] = True

        logits, val = self.forward(inp, act_pos, mask)
        probs = F.softmax(logits[0], dim=-1)

        if temperature <= 1e-5:
            chosen_idx = int(torch.argmax(probs).item())
        else:
            scaled_logits = logits[0] / temperature
            scaled_probs = F.softmax(scaled_logits, dim=-1)
            chosen_idx = int(torch.multinomial(scaled_probs, 1).item())

        prob_dict = {a: float(probs[int(a)].item()) for a in legal}
        return Action(chosen_idx), prob_dict, float(val[0].item())
