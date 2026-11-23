# CI Doctor

Tier 2 Semantic-FP flagship application (`A5`) for clustering CI failures, bisecting regressions, and verifying candidate fixes.

## Highlights

- **Known-flake crisp fast-path (`pluggable`)**: Deterministic signature detection resolves routine network/runner flakes with 0 model calls.
- **Confidence-gated clustering (`semanticPartition`)**: Splits non-crisp traces into `flake`, `env`, and `regression` clusters while routing ambiguous logs to `uncertain`.
- **Well-founded bisection (`iterateOn` + `withMeasure`)**: Binary search over commit ranges strictly shrinks the candidate interval each step.
- **Assertion-preserving fix selection (`branchAndSelect`)**: Evaluates candidate patches in isolated `Folder` transactions and rejects any patch that fails tests or decreases `assert`/`expect` counts or introduces `.skip`.
