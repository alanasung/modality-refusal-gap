from __future__ import annotations

import numpy as np


def enrich_items(items, cfg):
    out=[]
    rng=np.random.default_rng(0)
    for i,row in enumerate(items):
        r=dict(row)
        r["modality"] = "vision" if i%2==0 else "text"
        r["pixel_stub"] = rng.random((8,8,3)).tolist()  # tiny image stub
        r["harmful_intent"] = int(row.get("label",0))
        out.append(r)
    return out

