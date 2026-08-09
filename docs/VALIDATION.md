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

## P6 rigor pass (behavioral patch + direction + hydra)

| Fix | Status |
|---|---|
| Behavioral patch refuse-rate delta (`refuse_rate_before/after`, `behavioral_delta`, `patch_score_mode`) | OK (`patching.py`, `layers.py`) |
| HF unknown image prefix stays `alignment_unresolved` (fail-closed) | OK |
| Deterministic matched_group parity split (no `__hash__`) | OK (`direction.py`) |
| Prefer refuse/comply fit; gate fallback via `direction_claims_ok` + `fit_rule` | OK |
| Holdout AUROC projection→behavior when labels allow | OK |
| Hydra cumulative top-k with `rerank_after_ablate=true` | OK (`hydra.py`) |
| Utility/VLSBench stand-ins cannot set `claims_utility=true` | OK (`utility.py`, `vlsbench.py`) |
| Local mini arithmetic stamped `status=local_mini` | OK |
| Hub-free P6 domain tests | OK (`tests/test_domain_p6_behavioral.py`) |

Residual (frontier / licensed corpora, not empty stages): real Mono-InternVL weights, licensed VLSBench root, full MMLU/MMBench harness.

## P7 rigor pass (stats + claim contracts)

| Fix | Status |
|---|---|
| Dual-measured contrast gate (`contrast_claim_ok` requires unified+modular `architectural_claim_answered` and non-synthetic backends) | OK (`architectures.py`) |
| Local leakless mini-fixture (PNG+JSON under `data/fixtures/leakless_mini/`, decoupled text/image intents) | OK |
| Gap measured on fixture; `corpus=local_leakless_mini`; `claims_utility=false` | OK (`vlsbench.py`) |
| Bootstrap CI on text−image refuse gap + `mde` / `null_claim` | OK (`gap.py`) |
| Pilot power honesty: modest `n_items=32`, stamp `power_status=micro` / `gap_claim_ok=false` when underpowered | OK (`pilot.yaml`, README) |
| Hub-free P7 domain tests | OK (`tests/test_domain_p7_contrast_gap.py`) |

Residual (unchanged): licensed VLSBench / Mono-InternVL weights for reportable measured contrast; powered gap floor still requires ≥24 harmful pairs per modality.

