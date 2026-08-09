# VALIDATION — vlm-refusal-circuit (P1 upgrade)

## Codex v1 (historical)
- Verdict: SERIOUS_PROBLEMS — empty stages / text-only model.

## Codex v2 (pre-P1)
- Verdict: PASS_WITH_NOTES — stages wired but modular+synthetic default.

## Codex P1 (post-upgrade, gpt-5.6-sol)
- Command: `codex exec -m gpt-5.6-sol -s read-only` via `orchestration/codex_gate.py validate vlm-refusal-circuit`
- Verdict: **SERIOUS_PROBLEMS**
- Summary: Engineering skeleton improved, but several measurements remain proxy-grade (VLSBench stand-in, utility proxies, direction = harmful−benign, hydra component ranking incomplete).
- Blocking themes: causal patching depth on HF VLMs, VLSBench corpus adapter, real MMLU/MMBench, refusal-vs-compliance direction fitting, model-family adapters.

## Grok P1 (`cursor-grok-4.5-high-fast`)
- Artifact: `orchestration/out/grok/validate/vlm-refusal-circuit.p1.md`
- Verdict: **PASS_WITH_NOTES**
- Summary: Unified-subject pilot + honesty rule + contrast/hydra stages fix the architectural deferral; remaining issues are proxy-depth measurements, not missing stages.

## Key fixes applied in P1
- `pilot.yaml`: `OpenGVLab/Mono-InternVL-2B`, `force_synthetic_vlm: false`, `architecture: unified`, `allow_modular_as_subject: false`
- `full.yaml`: `adept/fuyu-8b` + modular contrast
- `smoke.yaml` alone forces synthetic / non-reportable
- Honesty rule in `load_vlm`: refuse modular-as-subject; unanswered synthetic on unified load failure; optional `require_measured_vlm` hard-fail
- Real activation capture (`activations.py`), no random residuals; gap errors not modality-fabricated
- Sequential contrast measurement (memory-safe)
- Stages: `contrast` + `hydra` added
- Dead generic pipeline quarantined

## Reconciliation
Codex SERIOUS_PROBLEMS vs Grok PASS_WITH_NOTES. **Split judgment:**
- **Architectural / alignment engineering:** PASS (Grok + our tests).
- **Measured scientific validity on real weights:** still open (Codex) until Mono-InternVL/Fuyu weights are cached, VLSBench corpus is local, and utility subsets are real.

Residual blockers for a prior work-grade measured pilot (not engineering scaffolding):
1. Download and pin Mono-InternVL (or Fuyu) immutable revision; set `require_measured_vlm: true` for reportable runs.
2. Provide `VLSBENCH_ROOT` with a versioned leakless manifest.
3. Fit refusal direction from refusal/compliance contrasts, not only harmful/benign.
4. Wire real MMLU/MMBench subsets for the full profile.

Operating judgment: **ship P1 architectural upgrade**; treat Codex science-depth items as the measured-pilot backlog before claiming results.

## P5 rigor pass (measured prior work-critical paths)

- Live / measured paths preferred; synthetic remains smoke-only with honesty stamps.
- Claim gating tightened where proxies previously looked like evidence.
- Domain tests green without Hub downloads.

