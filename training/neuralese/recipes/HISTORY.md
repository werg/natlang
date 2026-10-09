# Recipe history

Experiment history moved out of `description` for recipes that use `extends`/`overrides` (plans/ARCHITECTURE_IMPROVEMENT.md C3). Each entry is the original description, verbatim. Base: `foundation-maple-v1`.

## token-preserving-foundation-maple-c12-ctx512

foundation-maple-v1 at cutoff 12 with 512 context windows (v1 at cutoff 12 with 128 overfit: train KL 0.22, held KL 1.24 / agreement 0.54). 1,024 windows peak ~51 GB on unified memory, more than fits beside the teacher.

## token-preserving-foundation-maple-c12-ctx1k

foundation-maple-v1 at cutoff 12 with 1,024 context windows (v1 at cutoff 12 with 128 overfit: train KL 0.22, held KL 1.24 / agreement 0.54).

## token-preserving-foundation-maple-c18

foundation-maple-v1 at cutoff 18 (128 context windows; fits beside the teacher). Cutoff 12 plateaued at held agreement 0.54.

## token-preserving-foundation-maple-c18-ctx512

foundation-maple-v1 at cutoff 18 with 512 context windows (v1 at cutoff 12 with 128 overfit: train KL 0.22, held KL 1.24 / agreement 0.54). 1,024 windows peak ~51 GB on unified memory, more than fits beside the teacher.

## token-preserving-foundation-maple-c18-ctx1k

foundation-maple-v1 at cutoff 18 with 1,024 context windows (v1 at cutoff 12 with 128 overfit: train KL 0.22, held KL 1.24 / agreement 0.54).

## token-preserving-foundation-maple-c21-ctx512

foundation-maple at cutoff 21 with 512 context windows (cutoff 18: held agreement ~0.64 with 512 windows vs 0.61 with 128; depth helps more than data).

## token-preserving-foundation-maple-c23-ctx512

foundation-maple at cutoff 23 with 512 context windows (held best agreement/KL: c12 0.54/1.24, c18 0.646/0.748, c21 0.789/0.283 still improving at step 8064; depth helps most).

## token-preserving-foundation-maple-c23-ctx1k

foundation-maple at cutoff 23 with 1,024 context windows (c23 with 512 plateaued at held agreement 0.854 / KL 0.126 from ~4k steps: KL passes, agreement does not; more data next).

## token-preserving-foundation-maple-c23-calibrated-v1

foundation-maple-c23-ctx512 with the agreement bound calibrated for Maple (owner 2026-10-06: no bug -> move on). Diagnostic of c23-ctx512 (runs/maple-foundation-20261005/diagnostics/gate-diagnostic-c23.json): the readout is exact at initialization; agreement is 99.6-100% where Maple puts >=0.9 on its top token and 63-66% below 0.3; the projection's token is in Maple's top 5 at 98.6-98.8%; ~3% of positions are exact bf16 ties. Top-1 agreement on Maple is bounded by its own next-token entropy (source positions: 43% below 0.3), so the bound is 0.75 per stratum; KL keeps 0.25.
