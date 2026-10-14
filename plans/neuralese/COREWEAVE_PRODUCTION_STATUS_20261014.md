# CoreWeave production state, 2026-10-14 19:05 UTC

Two owner-authorized full-weight shared production recipes were released at
18:49 UTC. This is current execution evidence, not stage qualification.

| Line | Sandbox | Actual GPU | Host request | Output |
|---|---|---|---|---|
| LFM QAD R8 PH continuation | `7ee701ae-8e56-49d8-b97a-594f7eb33fe9` | H100 80 GB, sm90 | 8 CPU, 64 GiB | `/work/runs/lfm-r8-ph-production-20261014/direct_sft` |
| Mellum fresh R8 warmup | `a7858948-0e1a-4c89-bb69-e2cc29cbc243` | RTX PRO 6000 Blackwell 96 GB, sm120 | 4 CPU, 64 GiB | `/work/runs/mellum-r8-fresh-production-20261014/direct_warmup` |

Both use no-volume root storage and NGC `nvcr.io/nvidia/pytorch:26.09-py3`.
Requested tag and official amd64 manifest SHA
`f71fd67f6d71f685e0bd08ce6c8df48807624cc5d48886370d3471bc4f89d81d`
are separately recorded in launch receipts. Actual runtime: Torch
`2.14.0a0+b2c75dd062.nv26.09`, CUDA 13.4, Transformers 5.19.0.
The LFM root disk is approximately 30.7 TB; Mellum approximately 7.68 TB.

LFM uses recipe `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1.json`
(commit `225b60a7`) with `--handoff`, exact qualified Pop warmup410 full
checkpoint SHA `13ebca2a4c2f2f087ce2cfc4a635114b4aa2b91a017714958bed71cd33551761`,
its optimizer/RNG, certificate/identity/monitor reference and clean raw heads.
All five handoff artifacts were SHA-verified in the sandbox. Protected PH prefix
SHA `60c69c269ac7cfc6ee26cbc5d5c78c2d72e98d14ed8370027fe9e800b5c3edf8`
was privately transferred and independently SHA-verified. Its full data scan
reached 90.15% at 19:03 UTC. First update/restore qualification remains pending. At19:25 the final merged-row
window scan was62.79% complete; startup is three streamed passes (validation,
dedup plus merge metadata, merged windows), not GPU work.

Mellum uses separately declared FRESH recipe
`neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1.json`
(commit `2cfa56ea2b`), identical pinned clean base/raw heads/admitted R8 full
warmup inputs. This is not a DGX checkpoint/optimizer resume. At 19:04 UTC
94 updates completed without errors. Step-0 identity reports passed=true and
logits_equal_base=true. Example update94: 3.651 seconds, 29,745 forward tokens,
CUDA peak allocated 55.25 GiB/reserved61.06 GiB. GPU utilization98–100%.
Actual full-context packed windows reach32767 tokens with checkpointing.
Planner uses actual101,975,851,008-byte device capacity. Actual runtime installs
28 FP8 expert modules using `triton-grouped-experts-v1` on sm120 and168 MuonSR
latents. These are real trainer evidence; additional optimization claims require
measured dispatch/runtime receipts. Production schedule is4851 updates at0.5epoch,
checkpoint cadence170 and task/CE evaluation400.

## Durability and lifecycle

Private project `werg/natlang` access was explicitly verified PRIVATE before
upload. PH transferred as immutable private artifact; no owner credential enters
the sandbox. A faster cloud export path uses official W&B CreateRunFiles to
issue an exact-file presigned PUT URL on DGX, then sends only that capability
and required headers to the sandbox. Actual Mellum identity168bytes was uploaded
HTTP200 in0.273s and independently retrieved with exact SHA
`e1059804617edeb679d1ffa62ea8778057c59df05c73985aa09ab1fd2bdc56b1`.
Private export run `werg/natlang/ywcj2mvb`. Full checkpoint export is pending;
small identity export does not establish full model/optimizer durability.
A ledgered watcher awaits the first atomic Mellum checkpoint, makes an immutable
root-disk copy, then privately uploads and independently hashes the fetched
object from cloud. This avoids a47GiB DGX root allocation and slow gateway transfer.

Manual release mistakenly exec'd the recipe as PID1 rather than the existing
bootstrap hold wrapper. PID1 exit can end the sandbox and file API. SIGSTOP was
attempted but verified INEFFECTIVE (PID1 remainsState=S); do not assume held.
Current recipe PID1, actual LFM trainer9887, Mellum trainer5860. Stop requests
must target the exact active trainer and wait for full atomic state commit and
durable export before SDK stop. Ordinary trainer SIGSTOP can hold fileAPI alive
before terminal exit if required; it has not been applied. Subsequent launches
must use existing `scripts/coreweave/bootstrap.sh` API-transport hold/collect
mechanism. Do not stop either pod or infer root-disk persistence after termination.
Parent retains local DGX Mellum until steady cloud training AND full checkpoint
save/export are verified; local job lifecycle belongs to the parent.

Exact private launch, progress, transfer and export receipts are under
`/home/werg/data/dgxops/coreweave-production-*20261014.json`.
Signed capabilities and W&B runtime logs remain outside the checkout, mode600
or private directories; never commit them. D40 CPU conversion shares LFM root
disk in a frozen runtime and is parent-owned. Both production allocations remain
running; all earlier seven sandbox diagnostics were independently verified stopped.

## Reader cardinality proof, 19:24 UTC

Raw656263 rows become656262 reader rows through one existing exact-key duplicate,
not missing data or a held collision. Cache skip set is exactly{250111}
(zero-based). Earliest matching row46536 is retained. Both are train,
agentic-tool-loop, source groups
`swe-rebench-instance:Clinical-Genomics__patientMatcher-75` and
`swe-rebench-repo:Clinical-Genomics/patientMatcher`.
Kept ID `harness:openhands-v0.54:chatcmpl-87de9c62149901120aee88c6287d0bba:turn:0005`;
duplicate `harness:openhands-v0.54:chatcmpl-ea839420aa7d4dbdfb1c2a5dcb079e23:turn:0005`.
Full duplicate key: split=train,
text SHA `a59f7e173070e6ebe6195b5382fdb56a735dcf49cac69a08bf8e0af8188a40b0`,
token IDs SHA `9dcaa0cb390bb2fabaa45e0ecc88e6f5a65ea1d88447f2e629f20d7d6960a561`,
supervised suffix1658, sampling_group=null. Entire corpus scan found exactly
these two key matches and zero held same-text lines. Diagnostic terminal0;
receipt `/home/werg/data/dgxops/coreweave-production-lfm-reader-dedup-proof-20261014.json`.
No admission/training change required.

Mellum first atomic full state at170 committed19:13:25,
46,821,474,583bytes. Immutable export copy completed138seconds later;
SHA `93737f4e87454a3f07c642bec2fe36e5d9fcb8fa0b8e8027d4999c363f442381`.
Actual private upload endpoint is `storage.googleapis.com`; GCS supports5TiB
objects across uploadmethods ([official quotas](https://docs.cloud.google.com/storage/quotas?hl=en)).
At19:25 scoped PUT process had read~22GB/46.8GB in370seconds (~59MB/s),
with24MB RSS, while actual trainer continued326updates/GPU100%.
HTTP completion and independent remote SHA verification remain pending.
