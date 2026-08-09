from __future__ import annotations
def fit_extra(cfg, run_dir, x, y, prob):
    return {"refusal_direction_dim": int(x.shape[1]), "cross_modal_patching": True}

