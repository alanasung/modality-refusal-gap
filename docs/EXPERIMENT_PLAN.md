# Experiment plan

Stage-by-stage design. Each stage is registered in `src/vlmrefusal/stages.py`
and appears in `python -m vlmrefusal stages`.

## Stages

| stage | responsibility |
|---|---|
| `render` | typographic rendering of text prompts into matched images |
| `matched` | matched harmful/benign, text/image item construction |
| `ocr_check` | transcription gate so OCR failure is not read as refusal |
| `direction` | refusal-direction extraction from text contrastive pairs |
| `layers` | layer-wise projection and cross-modal activation patching |
| `vlsbench` | VLSBench leakless evaluation arm |
| `utility` | MMBench general-capability cost of any steering fix |

## Execution order

Stages form a linear dependency chain by default; the runner resolves the order
topologically, so a stage may be run alone and its prerequisites are pulled in
automatically:

```bash
python -m vlmrefusal run -c configs/pilot.yaml --stage utility
```

## Controls and their purpose

- If the model simply cannot read the rendered text, a low refusal rate is an OCR failure, not a safety gap. A transcription check gates every item.
- Typographic rendering is visually encoded text, which is a narrow slice of visual harmful intent. Findings must not be generalized to the whole visual channel, which is why the VLSBench leakless arm is mandatory rather than optional.
- A modular VLM is a BASELINE for architectural contrast, never a substitute. If no unified encoder-free model can be run, the central architectural question is unanswered and the writeup must say so plainly instead of reporting modular results as if they settled it.

## Decision rules

Report effect sizes with bootstrap intervals. Treat an interval that spans zero as a null result and report it as such; do not reach for a subgroup that reaches significance.

## Reproducibility

Every run records a manifest with the git sha, a config fingerprint, resolved
device and dtype, package versions, per-stage timings, and metrics. Seeds are
set across python, numpy, and torch. Known determinism limits are recorded in
the manifest rather than assumed away: MPS does not support
`torch.use_deterministic_algorithms`, so small numeric drift between runs is
expected and should not be read as an effect.

## Scale

The pilot profile is what actually runs on the target machine. The full profile
describes the intended scaled-up run. When reporting any result, state which
profile produced it; a pilot-scale null is weaker evidence than a full-scale
null and the writeup must not blur them.
