"""
Тренер glyph-level LM. Универсальный — обучает любую модель (Войнич или контроль)
по потоку id. Логирует loss в logs/<name>_train.jsonl, чекпоинт в checkpoints/.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from voynich_lm.model import GlyphLM, default_config
from voynich_lm.data import GlyphTokenizer, make_batches, evaluate_loader


def cosine_lr(step, total, warmup, lr_max):
    if step < warmup:
        return lr_max * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return lr_max * 0.5 * (1 + math.cos(math.pi * prog))


def train_model(ids: list[int],
                tok: GlyphTokenizer,
                name: str,
                ctx: int = 256,
                batch_size: int = 64,
                n_steps: int = 4000,
                lr_max: float = 1e-3,
                warmup: int = 200,
                weight_decay: float = 0.1,
                grad_clip: float = 1.0,
                eval_every: int = 250,
                val_ids: list[int] | None = None,
                seed: int = 7,
                device: str | None = None,
                log_dir: Path = Path("logs"),
                ckpt_dir: Path = Path("checkpoints"),
                cfg: dict | None = None,
                verbose: bool = True) -> tuple[GlyphLM, dict]:
    """
    Обучает модель на ids. val_ids — отложенный поток (для per-step val loss).
    Возвращает (model, history). Сохраняет чекпоинт checkpoints/<name>.pt.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = cfg or default_config()
    cfg["max_ctx"] = max(cfg.get("max_ctx", ctx), ctx)
    model = GlyphLM(tok.vocab_size, cfg).to(device)
    n_params = model.n_params()
    if verbose:
        print(f"[{name}] параметров: {n_params/1e6:.2f}M | устройство: {device} | "
              f"шагов: {n_steps} | ctx: {ctx} | batch: {batch_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay,
                            betas=(0.9, 0.95))

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logf = log_dir / f"{name}_train.jsonl"

    # подготовим val-батчи для честной оценки
    if val_ids is not None and len(val_ids) > ctx + 1:
        VX, VY = evaluate_loader(val_ids, ctx, batch_size, device)
    else:
        VX, VY = None, None

    history = {"name": name, "n_params": n_params, "n_steps": n_steps,
               "ctx": ctx, "vocab_size": tok.vocab_size, "log": []}
    t0 = time.time()
    model.train()
    best_val = float("inf")
    best_state = None          # early-stopping: лучший state_dict по val_loss
    best_step = 0
    for step in range(1, n_steps + 1):
        # lr schedule
        lr = cosine_lr(step, n_steps, warmup, lr_max)
        for pg in opt.param_groups:
            pg["lr"] = lr

        X, Y = make_batches(ids, ctx, batch_size, device, n_batches=1, seed=step)
        logits, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step % eval_every == 0 or step == n_steps:
            vloss = None
            if VX is not None:
                model.eval()
                with torch.no_grad():
                    vl = []
                    for i in range(0, VX.shape[0], batch_size):
                        _, l = model(VX[i:i+batch_size], VY[i:i+batch_size])
                        vl.append(l.item() * (VY[i:i+batch_size] != 0).sum().item())
                    ntok = sum((VY[i:i+batch_size] != 0).sum().item()
                               for i in range(0, VX.shape[0], batch_size))
                    vloss = sum(vl) / max(ntok, 1)
                model.train()
            rec = {"step": step, "train_loss": loss.item(), "lr": lr,
                   "val_loss": vloss, "dt": time.time() - t0}
            history["log"].append(rec)
            logf.open("a").write(json.dumps(rec) + "\n")
            # early-stopping checkpoint: запоминаем лучший по val
            if vloss is not None and vloss < best_val:
                best_val = vloss
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                vl_str = f" val={vloss:.4f}" if vloss else ""
                star = " *" if (vloss is not None and vloss == best_val) else ""
                print(f"  [{name}] step {step:5d}/{n_steps}  train={loss.item():.4f}{vl_str}{star}  "
                      f"lr={lr:.2e}  ({time.time()-t0:.0f}s)")

    # восстанавливаем лучший (early-stopping) state для чекпоинта
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    history["best_val_loss"] = best_val
    history["best_step"] = best_step
    ckpt = {
        "model_state": model.state_dict(),
        "cfg": cfg,
        "vocab": tok.to_json(),
        "name": name,
        "n_params": n_params,
        "best_val_loss": best_val,
        "best_step": best_step,
    }
    torch.save(ckpt, ckpt_dir / f"{name}.pt")
    if verbose:
        print(f"[{name}] сохранён чекпоинт (best val={best_val:.4f} @step {best_step}): "
              f"{ckpt_dir / (name+'.pt')}")
    return model, history


def load_model(name: str, ckpt_dir: Path = Path("checkpoints"),
               device: str | None = None) -> tuple[GlyphLM, GlyphTokenizer]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_dir / f"{name}.pt", map_location=device, weights_only=False)
    tok = GlyphTokenizer.from_json(ckpt["vocab"])
    model = GlyphLM(tok.vocab_size, ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok
