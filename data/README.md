# Data

This directory holds **tiny checked-in manifests and Hub-free fixtures**. Raw
corpora, embeddings, and result tables are gitignored.

## Contract

| Path | Committed? | Purpose |
|---|---|---|
| `data/README.md` | yes | This file |
| `data/manifests/*.json` | yes if < 2 KB | Dataset provenance (seed, n, version) |
| `data/fixtures/leakless_mini/` | yes | Local leakless PNG+JSON mini-fixture (not VLSBench) |
| `data/raw/` | never | Downstream download target |
| `data/processed/` | never | Split caches |

## How to obtain data

1. Prefer synthetic pilot items via `scripts/build_dataset.py data=synthetic`.
2. For external corpora, document the exact URL, license, and revision in the
   manifest and never commit the rows.
3. Domain pilot `n_items` may be modest with `power_status=micro` /
   `gap_claim_ok=false`; do not treat that as a powered 512-item claim.
4. Leakless visual-channel smoke uses `data/fixtures/leakless_mini/`
   (`corpus=local_leakless_mini`, `claims_utility=false`). Licensed VLSBench
   still requires `VLSBENCH_ROOT`.

## Honesty

Anything produced from synthetic builders must be labelled `is_synthetic: true`
in result payloads and must never appear in a measured-results table. The local
leakless mini fixture is not a licensed benchmark and never sets
`claims_utility=true`.
