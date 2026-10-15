# SearchSwarm/OWL CPU preparation on the existing LFM sandbox

Policy is this natural-language procedure: use two spare host CPUs at nice19
only while real LFM host headroom permits it; preserve training/export priority;
fetch exact public pins, convert with shared adapters, protected-finalize, then
replay existing native call programs. No GPU model job or extra sandbox. Copies,
conversion and successful replay do not replace protected admission/registration.

## Decomposition

1. Check resources and dependencies with crisp external-service probes; choose
   two actually permitted CPUs from the reported affinity set.
2. Stage a frozen Git archive and exact protected/reservation data files.
   Git is the source transport; source trees are not copied between machines.
3. Fetch the two declared public dataset revisions and verify full SHA/size;
   this is existing acquisition plumbing and records license provenance.
4. Invoke existing `convert-agents` adapters and streaming `finalize`, retaining
   split groups, upstream sessions, rejected rows and all protected hits.
5. Invoke the existing shared `port_record_calls.sh` pipeline with one builder
   and one replayer in the isolated snapshot. Its repository root and builder
   count become explicit environment settings; canonical defaults stay intact.
   This is external execution plumbing, not a second native converter.
6. Preserve terminal receipts, manifests, omissions and compressed native turns;
   report to the supervisor for exact checks and later rendering/admission.

## Measured preparation

LFM sandbox7ee701ae-8e56-49d8-b97a-594f7eb33fe9 at22:28UTC had
16,426,283,008hostbytes used out of64GiB, zero memory events/OOM and29.86TB free
root storage. Allowed CPU affinity0..8 supports choosing7,8. Public source pins:
SearchSwarm/SearchSwarm-SFT4bad3397be6144a0fbbaac3121929e082f32865e and
camel-ai/OWL-SFT878b41ca2eb68d2247145f303a854adedd3f2fb3,
2,128,554,844bytes total. HuggingFaceHub, PyArrow, jsonschema and Torch exist;
Node24 and zstd CLI exist. Python zstandard is absent; these stages use the
existing zstd CLI and do not require that module. Protected index is missing on
LFM and must be staged with SHA
585b21e7091adfbc4f851ef35b0b3c6da00f032fffe69a8b98816a567abad7bf.
The existing Node dependencies are present; compile the isolated Git snapshot's
own TypeScript dist without changing the active trainer source/runtime.

## Actual staged job

The isolated snapshot is Gitddbebe9a121ccfe4c2db7ae68b45e2f45f476611,
61,109,391bytes, SHA
b36dd8ece8324a77bf60beb379dde94d9fc7aa1ad330173686b399537e2b0f84.
Runtime `/work/cpu-runtime/search-owl-ddbebe9a121c`; job
`/work/datajobs/search-owl-native`; owner spec
`/home/werg/data/dgxops/coreweave-search-owl-cpu-spec-20261014.json`.
Protected index and JeFF reservations were remotely SHA-verified before launch;
reservations SHA2ec7845cd19d9b42c504f2ac7535c3bc23510b913b03f5b76b6da911fd2b84c7.

First isolated build failed to resolve hoisted Node modules: the snapshot needed
both root and ts-host dependency directories. Linking both existing dependency
locations fixed the build; no active source/runtime was changed. Both public
source pins then downloaded and passed size/full-SHA checks. First conversion
stopped because the shared adapter's relative data alias was absent from a
Git-only snapshot; adding that isolated alias to the actual pinned raw root
fixed the binding. These setup failures did not produce admitted data.

At22:40UTC, actual runner28555/converter28671 are alive, exit receipt absent,
and `convert-agents searchswarm owl-sft` is running. Existing native shell's
EXIT cleanup signals its process group, so the external task launcher gives
phases independent sessions and records each phase PID. This preserves the
shell's existing child cleanup while keeping the outer durable exit receipt.
The generic submit helper refuses duplicate live workers and clears an obsolete
exit marker only after verifying the prior worker is gone. Poll reads are
bounded to64KiB; old setup traces in cumulative logs are historical, not newly
failed conversion results. Closure, native replay and admission remain pending.

## Prepared reuse for Dolci and Cognitive Kernel, 23:35UTC

After Search/OWL completes its current protected-finalization and existing native
pipeline, reuse the same isolated spare-CPU procedure for D30 and D34. Do not
run a second converter concurrently with it or change either training recipe.
The prepared owner-side spec is
`/home/werg/data/dgxops/coreweave-dolci-cognitive-cpu-spec-prepared-20261014.json`;
its existing staging-helper derivative writes an exact Git-revision runtime and
executable spec before submission. It includes both root and ts-host dependency
links, the required relative data alias, independent phase sessions and durable
phase PID receipts from the resolved Search/OWL setup. No new conversion,
closure, native builder or replay implementation is introduced.

Fetch `allenai/Dolci-Instruct-SFT-Tool-Use` at
`dc042846f0f2de0f15eedae3d6ced04223ed47eb` and
`CognitiveKernel/CognitiveKernel-Pro-SFT` at
`6443666c406f863898b0924cf1a3e1a2512180e8`, exactly1,663,608,164catalog bytes.
Verify every file's size/full SHA and available LFS SHA. Run shared
`convert-agents --corpora dolci-tool-use cognitive-kernel`, then streaming
`finalize` with the same protected benchmark binding, followed by the existing
one-worker native pipeline under the renewed host-headroom check. The local
644,471candidate count is a comparison reference, not a predeclared cloud
result or training admission. Preserve source license cards, split groups,
closure omissions and exact native receipts. The supervisor registers successful
source/native outputs and admits rendering/union inputs separately.

## Closed source and resumed native replay, 2026-10-15 00:13UTC

Protected finalization completed170,272train records with no exact duplicates,
invalid records, protected hits or closure violations; all-pass input hashes and
stat bindings matched. Native build committed163,270items in40,818cases/
164shards. It explicitly retains7,002omission IDs with reason
`evidence-over-cap-needs-paged-program`; these need a later paged-program
conversion, rather than being lost or counted as native training decisions.

The supervisor approved two replayers on the same already reserved affinity7,8,
nice19 and zero CUDA. A driver-only TERM was deferred by Bash while its
foreground child ran; the exact stop verifier refused premature relaunch. After
verifying CPU-only PGID41120 and the separate trainer PGID35661, scoped group
TERM completed the existing cleanup. Committed summary shards00000–00002,
all164IRshards and frozen runtime/page bindings were retained. The resumable
native driver restarted at00:12:25 with WORKERS2, runner46765/driver46769.
Concrete argv and verified frontier are in dgxops
`coreweave-search-owl-cpu-spec-20261014.json` and
`coreweave-search-owl-two-worker-resume-frontier-20261015.json`.

Only the existing native driver reruns; source conversion/finalization do not.
Both phase launcher and driver have separate sessions so the shell's EXIT
cleanup cannot destroy the outer terminal receipt. Polling checks actual family
build manifests and completed shard summaries; a zero/zero shell exit does not
establish native output success. At00:13, LFM still trained31,743forward tokens
in0.905seconds with33,482supervised tokens/second; anon9.23GB and OOM0.
Monitor matched update throughput and anon pressure, reducing back to one
replayer if measured competition appears. D30/D34 waits until this entire native
pipeline is terminal and its artifacts survive; no second converter runs beside it.

### Owner-host private durability (2026-10-15)

The owner selects completed source archives for offload. Reuse the private
run-file exporter with an owner-host input option: exact file fingerprint and
streaming SHA, the same private-project check and GCS scoped PUT, followed by
an independent complete GET/SHA readback. The crisp additions are external
service transport and exact byte verification. They decide neither source
quality nor training admission. Account credentials remain on the owner host;
artifact availability and admission remain separate. No second account route
or cloud converter is introduced. Keep the original archive until its assigned
durable-readback consumer completes.

### BuildAndVerifyConvolution(source, imageABI, device, referenceInputs)

The supervisor chooses the exact registered source and architecture.
1. Read the pinned source commit and existing Pop patch/build manifest.
2. Build the corresponding sm90 wheel against the active Torch2.14/CUDA13.4
   image in an isolated CPU-only build directory, using available spare CPU.
3. Keep the frozen forensic runtime unchanged. After its GPU evaluation ends,
   reuse the matched forward/backward and throughput harness against the exact
   reference inputs on the same H100.
4. Record source, patches, wheel/build hashes, numeric differences and timing.
   The supervisor admits an architecture mapping only after those verifiers
   pass; building a wheel alone does not qualify it.

Crisp work is public Git/source transport, compiler invocation, ABI/device
identity, hashing and exact numerical/throughput measurements. It neither
chooses a model policy nor changes training or evaluation controls.

### PackClosedNativeForDurability(closed, native, receipts)

The owner selects the completed Search/OWL trees. Preserve every closed handoff,
all native IR/graph/turn outputs, omissions and conversion/replay receipts.
Package closed and native independently with the existing mined-source archive
protocol, verify every decoded member against its original byte hash, then use
the existing private scoped exporter and independent full-object readback.
Availability grants no training admission; source/held/quality checks remain
explicit. The shared `scripts/coreweave/source_archive.py` promotes the existing
operator pack/verify implementation with explicit source/spool/name arguments.
Crisp code is exact byte transport, atomic commit and verifiers. It makes no
source-selection, conversion or promotion decision. Legacy local wrappers should
call that shared implementation rather than retaining another copy.
