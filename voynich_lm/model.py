"""
nanoGPT-декодер, glyph-level. Causal self-attention + RoPE.
~1.5M параметров при конфиге по умолчанию (d=128, h=4, l=4).

Не зависит от kv-cache (модель мала). Расчёт attention — через SDPA из torch.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    """Rotary Position Embedding. Вынесен отдельно для прозрачности."""
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
        cos = freqs.cos()[None, :, None, :]         # (1, T, 1, hd/2)
        sin = freqs.sin()[None, :, None, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def rotate(self, q, k):
        # q,k: (B, T, H, hd)
        T = q.shape[1]
        cos = self.cos_cached[:, :T]               # (1, T, 1, hd/2)
        sin = self.sin_cached[:, :T]
        cos = torch.cat([cos, cos], dim=-1)         # (1, T, 1, hd)
        sin = torch.cat([sin, sin], dim=-1)
        q2 = (q * cos) + (self.rotate_half(q) * sin)
        k2 = (k * cos) + (self.rotate_half(k) * sin)
        return q2, k2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.h = cfg["n_heads"]
        self.hd = d // self.h
        assert d % self.h == 0
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = cfg.get("dropout", 0.1)
        self.rope = RoPE(self.hd, max_seq_len=cfg.get("max_ctx", 1024))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.hd)
        q, k, v = qkv.unbind(dim=2)                # (B,T,H,hd)
        q, k = self.rope.rotate(q, k)
        # к (B, H, T, hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0,
                                            is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["d_model"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["d_model"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GlyphLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}
        cfg.setdefault("d_model", 128)
        cfg.setdefault("n_heads", 4)
        cfg.setdefault("n_layers", 4)
        cfg.setdefault("dropout", 0.1)
        cfg.setdefault("max_ctx", 512)
        self.cfg = cfg
        d = cfg["d_model"]
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])
        self.ln_f = nn.LayerNorm(d)
        # weight-tied head
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, return_hidden: bool = False):
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=0)
        if return_hidden:
            return logits, loss, x  # скрытые состояния (B,T,d)
        return logits, loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def nll_per_position(self, idx, targets):
        """NLL (натуральный логарифм) по каждой позиции — для position/perplexity анализа."""
        logits, _ = self(idx, targets=None)
        lp = F.log_softmax(logits.float(), dim=-1)
        tgt = targets.unsqueeze(-1)
        nll = -lp.gather(-1, tgt).squeeze(-1)   # (B,T)
        # маска pad
        mask = (targets != 0).float()
        return nll, mask


def default_config() -> dict:
    return {"d_model": 128, "n_heads": 4, "n_layers": 4, "dropout": 0.1, "max_ctx": 512}
