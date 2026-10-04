# DGX screen recovery audit

Read-only status and prepared-input audit. I made no provider calls, service restarts, or collection launches. After the retry inputs were reviewed, root launched the two support screening services from the v2 input directories.

The semantic and ContractNLI screen services exited with systemd success, but 87 selected rows have only socket/connection-reset errors and no `support_quality`, `support_passed`, `cases`, or `gates_passed` result. These are unscored transport failures, not negative model outcomes. The recovery packets contain only those exact original episodes, retaining their original JSONL bytes and order. Scored rows were not retried.

| Cohort | Screened | Transport retries | Details |
|---|---:|---:|---|
| Semantic headroom v2 | 112 | 75 | 58 socket closed; 17 ECONNRESET; 449 support cases, 374 groups; SciFact only |
| ContractNLI pilot v1 | 17 | 12 | 10 socket closed; 2 ECONNRESET; 72 support cases, 8 groups |

## Runtime and launch state

The outage-resilient runtime v6 was built from `0d19c05e`, at `/mnt/external/natlang-development-data/runs/dgx-development-generated/self-improvement-expansion-20261004/runtime-semantic-v6-0d19c05e`, with frozen manifest SHA-256 `fa22a6960d50941c29442f06df48932931d2c2843ec751f8fc25471c4e0e86bf`. Root launched the prepared support screens from the v2 input directories; I verified their units were active while writing the live batch plans. Existing ContractNLI collection remains separate and was left untouched.

## Prepared recovery inputs

- Semantic input: `/mnt/external/natlang-development-data/runs/dgx-development-generated/self-improvement-expansion-20261004/semantic-headroom-v2-recovery-transport-v2/episodes.jsonl` — SHA-256 `591c03e9ea1c2e185cb98c01007a9f01540d87b7fc095132ca73dec138704516`; IDs file SHA-256 `8b2fe95f9f1f615440e3e8525608632e9398087d0a2e9f2bf948de49b1062e20`; batch-plan SHA-256 `aba032e97e676d284d1264485cd136ab07f0aa0711f528ce23b28c612c699437`.
- ContractNLI input: `/mnt/external/natlang-development-data/runs/dgx-development-generated/self-improvement-expansion-20261004/contractnli-pilot-v1-recovery-transport-v2/episodes.jsonl` — SHA-256 `4da2ceb15b57e648d324a3000ee7c30bdd959e37aedbfdecd03e4133242f4859`; IDs file SHA-256 `0d361aeccf0d03fee7e2ae1500bd7e46ca825ad1b386ba7132b2f8d8adc9d99c`; batch-plan SHA-256 `f1c0ff54358dfbec0bf15e29cb9300d435c7abde6502251cab778305b02a7f77`.

Both recovery plans specify train/support-only screening, concurrency 8, family-probe 0 so every preselected row is tried, and the current Qwen endpoint/model. The root-created v3 recovery-plan sidecars bind the v6 runtime and are separate from the live v2 directories; the launched jobs continue reading the byte-identical v2 input files. The `batch-plan-live-v1` directories in those active v2 paths now contain fixed 8-ID input-order partitions and prepared collection registries. They are not launch records; root controls the waiter/collector launch.

## Runtime-bound recovery and live batch plans

Runtime-bound v3 sidecars (prepared, not launched) and the registry are under `/mnt/external/natlang-development-data/runs/dgx-development-generated/self-improvement-expansion-20261004/`. Registry SHA-256: `bfd1d179eda192c4d9f9f7a408adf2484fbe49f99cc90b1b7f9aeed0185f9ae8`. The actual screens were launched from the v2 input directories; their episode bytes exactly match v3 inputs.

- Semantic batch-plan manifest: `semantic-headroom-v2-recovery-transport-v2/batch-plan-live-v1/manifest.json`, SHA-256 `68e24836b11125ea5bc856d71d8c49ccf2f29f461dd663feef6e182f68520af9`. Collection registry SHA-256 `bd279fbca3baf436fa6f28fc4e4aeb598c80e18edec839af024265bd534628d4`. 75 IDs partitioned into 10 batches (eight each, final batch three).
- ContractNLI batch-plan manifest: `contractnli-pilot-v1-recovery-transport-v2/batch-plan-live-v1/manifest.json`, SHA-256 `e38175195f65d84bbbf7946db5a53b2280af74686264e23a41424f56c13aaea3`. Collection registry SHA-256 `d13f6a97eb07c6529c146951c078aa8976dd7ff02fb932c0187c048edb941389`. 12 IDs partitioned into two batches (eight and four).

Both plans pin input and v6 runtime hashes. Collection roots are `queue-semantic-recovery-v1/batch-NN` and `queue-contractnli-recovery-v1/batch-NN`, using the existing `collector-pool-semantic-v3` with two workers, three experiments, two ablations, and one attempt. No collection waiters were launched by this preparation.
