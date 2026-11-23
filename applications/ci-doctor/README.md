# CI Doctor

Tier 2 Semantic-FP flagship application (`A5`) for clustering CI failures, bisecting regressions, and verifying candidate fixes.

## Highlights

- **Known-flake crisp fast-path (`pluggable`)**: Deterministic signature detection resolves routine network/runner flakes with 0 model calls.
- **Confidence-gated classification (`semanticPartition`)**: Splits non-crisp traces into `flake`, `env`, and `regression` while routing ambiguous logs to `uncertain`; within each kind, `semanticMatch` clusters failures that share a root cause.
- **Well-founded bisection (`iterateOn` + `withMeasure`) with a natural-language stop**: binary search strictly shrinks the candidate interval each step; `bisectSettled.nl` stops it when a tested commit shows a different breakage or cannot be tested.
- **Assertion-preserving fix selection (`branchAndSelect`)**: Evaluates candidate patches in isolated `Folder` transactions and rejects any patch that fails tests or decreases `assert`/`expect` counts or introduces `.skip`.
