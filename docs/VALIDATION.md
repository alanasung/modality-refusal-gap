# VALIDATION — vlm-refusal-circuit

## Codex v1 (historical)
- Verdict: SERIOUS_PROBLEMS
- Summary: This is a tidy infrastructure skeleton, not a runnable research pilot: its stages are empty, its configured model cannot see images, and the core causal and measurement procedures remain undefined.

## Codex v2
- Verdict: PASS_WITH_NOTES
- Summary: Analogous to introspection-verbalization Codex v2: X1–X13 OK; stages implemented with a real `make pilot` path; synthetic/proxy pilot default; several model revisions still on `main`.
- KEY_FIXES_OK: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13

## Grok (dual-validate)
- Verdict: PASS_WITH_NOTES
- Summary: Domain stages (render→…→utility) call matched/OCR/direction/layers/VLSBench/utility; pilot names SmolVLM with SyntheticVLM fallback. X-fixes absorbed. Engineering PASS_WITH_NOTES; architectural unified-VLM gap is an alignment issue, not a missing-stage failure.

### Remaining
- `force_synthetic_vlm: true` is the pilot default; measured SmolVLM weights are optional.
- Pilot subject is modular SmolVLM-scale (or SyntheticVLM), not a named unified encoder-free checkpoint — engineering runnable, architectural claim deferred (see ALIGNMENT).
- Leftover generic `pipeline.py` is unused by `stages.py` (domain runners are wired); dead code hygiene only.

## Reconciliation
v1 text-only/empty-stage issues addressed via multimodal stage graph + SyntheticVLM/SmolVLM pilot. Grok engineering PASS_WITH_NOTES; do not confuse with alignment MATERIAL_DRIFT on unified architectures.
