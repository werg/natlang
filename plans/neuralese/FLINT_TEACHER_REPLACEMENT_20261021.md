# Flint generation teacher replacement

The owner requested an actual test of RecursiveSI/flint and replacement of Qwen
if decent. Flint is provisionally serving continuous verified generation in the
existing teacher sandbox. The full 48-case comparison is complete: Flint solved
32/48 (66.7%) and Qwen solved39/48 (81.3%). Flint had11 budget failures,2 invalid
actions and3 semantic mismatches. It solved3 cases Qwen failed, while its matched
task-success rate was lower.

On the 29 cases both solved, median Qwen/Flint elapsed time was2.94x. Counting
all successes and failures, summed case wall time was72,199.5s for Flint and
99,214.0s for Qwen. Verified successes per aggregate case-hour were1.60 versus
1.42, about13% higher for Flint. This is a practical proxy under different
serving loads, not isolated GPU throughput. The median on successful cases alone
overstates the aggregate benefit.

The operational reason to retain the provisional generator is faster verified
positive collection, not a claimed quality win. Admit only successful trajectories
that pass the existing full source, actor, held separation and conversion checks.
Diagnose concrete budget and semantic failures
before deciding on permanent replacement. The early 17/18 versus 15/18 observation
is superseded by these larger matched results; retain useful Qwen source data.

Both runs use the same source-bound task keys and seeds, the existing natlang
executor and full trajectory capture. No held tasks or external teacher generation
were added. Successful collection is separate from verified conversion, training
admission, learner adoption and actual exposure.

## Actual deployment

- Existing teacher sandbox: `78530f71-08fc-4f57-ace9-7fbad64aeeb8`, boot
  `6b9f17ce-9809-432f-a7cc-51ffac393cf2`, RTX PRO 6000 Blackwell, 96 GB.
- Base: [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), revision
  `2a247be0c0aec895ccb0ac7ca92229ec6df37288`.
- Quant: [official NVFP4 GGUF](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf),
  revision `6ec43e19b565fc8d560e2d04985d7724f20673bf`; actual artifact
  22,863,084,160 bytes, SHA-256
  `97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde`.
- Engine: [upstream Flint support](https://github.com/ggml-org/llama.cpp/pull/30928)
  at `1b44e30775dcb2f93952116b05f89e5bd797fb00`, merged with the upstream Jinja
  fix at `9febe63d5a379f8568bae7116a5b2ba844a4a4a6`; actual build
  `fe3e6ba71c885af4b58b4f1a53610d0f03acc716`, CUDA 13.4, sm120, Release.
- Eight serving slots, 65,536 tokens per slot, F16 KV, flash attention; temperature
  1, top-p .95, top-k 20, min-p 0. Actual reasoning content remains captured.
- Root: `/work/generation/flint-evaluation-20261021`; continuous generation under
  `production`, initial 64 TRAIN tasks, 16 collectors, then the existing broad
  qualified source sequence. No additional sandbox allocated.
- Serving intent SHA:
  `75b2a602263cd16281ed9eb49730883dc18dfbb65debc36ce7c7e467cba3c8ff`.
- Execution identity SHA:
  `b1a5ab2c050f3081c34ad710e7885b6190fe696023dbaaebd58aff62ac67d597`.
- Teacher source descriptor SHA:
  `5d6a0ee03aee5415ba8124f54c468d433f15ac6f6c57f055d78560018a22ce8f`.
- Guardian v4 preserves the collecting feeder and authenticates ordinary primary
  phase records separately from actual prefetched launches. Both producer and
  private publisher use the same protocol; 247 focused integration tests passed.
  Exact owner: PID 1771597, start tick 7039520. Actual generation GPU utilization
  was 99% at 17:59 UTC.

## Continuation obligations

The comparison is complete. Investigate the budget and semantic failures, and
measure useful production yield as the task mix broadens.
Maintain generation, verify private exports, convert and admit useful complete
trajectories through the shared mechanism for both students. Use the explicit Flint
namespace while retaining verified Qwen predecessors for deduplication. Do not
relabel existing Qwen batches or claim Flint exposure before actual adoption and
per-row training. Qwen batches 88–99 retain useful results and exact unfinished
accounting; reconcile their entire export inventory, rather than stopping at 88.

Increase serving concurrency or change other serving settings when measured useful
throughput supports it, preserving live cases at a natural boundary. New model
identity and sampling are immutable source configuration, not a separate data or
training pipeline. Shared decomposition:
`shared-state/TEACHER_MODEL_HANDOVER_DECOMPOSITION_20261021.md`.

Host receipts and operational controls are under `/home/werg/data/dgxops`:
`flint-continuous-native-teacher-activation-20261021.json` and
`flint-continuous-native-publisher-20261021/`. The publisher uses the explicit
CoreWeave SDK Python, not the project-only environment. Its durable spool is
`/mnt/external/natlang-development-data/teacher-incremental-flint-7853-20261021`.

## Verified integration at 18:49 UTC

The first two private-export tranches have passed source/oracle, exact actor,
TRAIN/held separation, paired actual-tokenizer reconstruction and full action
accounting. They contain 29 successful complete source trajectories and 179
captured actions, yielding 155 new ordinary SFT graphs for EACH student after
explicit omissions and deduplication. Data-only registration commits are
`cb137edae1adf4287aa4c59de98d524bbd2b155f` and
`431797596f407e89ea5e69fe7a2f56beb29dfce2`. Registration is separate from learner
adoption and actual exposure, which remains zero in these admission receipts.
The general continuous drain processes further Flint exports and Qwen batches
89–99 through the shared converters, strict admission and immutable manifests.
Qwen batch89/tr0 is also registered: 11 trajectories, 138 captured actions,
126 new graphs per student, commit `7f5ba3ec2863b4eb2d64145e99c929be4b7c93d0`.

Historical captures remain unchanged. A source-bound metadata continuation
attests the original independent-teacher role in the legacy converter field and
retains the original exported collection, queue, tranche frontier and actor.
Shared converter `35d05f7dd` resolves the serving alias from the exact transferred
teacher descriptor and checks its model/revision metadata, execution identity
and actor. Captured Flint provenance stays `flint`; it is never renamed to the
old `pennyroyal` alias. 191 focused tests passed, including actual per-batch
source/row validation and descriptor/actor tamper rejection. The normal global
source closure and its complete bound artifact checks passed with unchanged
source-protection implementation identities.

The complete comparison above supersedes all early partial success-rate claims.
The writer-occurrence correction `12a22fa1e` is integrated; its physical candidate
runtime is undergoing the required live model-facing measurement. The integrated
full check for `09bebe8316` passed at 20:18 UTC, independently of the unfinished
runtime measurement.

## Current integration and validation

The first mixed 481-graph continuation (six Flint tranches plus Qwen batch89
tranches0–1) was actually adopted by Mellum at step27732. LFM adopted that wave
alongside the preceding ready source additions at24663. Both learners then adopted
the second mixed wave: Mellum added1576 graphs at28028 and LFM added1584 at24750.
Actual active totals are1460605 TRAIN/25968 held for Mellum and1545644 TRAIN/25955
held for LFM. These totals are source obligations, not a claim of complete exposure.
The shared append mechanism preserves existing optimizer/RNG and per-row history.

The writer-occurrence candidate runtime is running the 48-task live measurement
on the same Flint actor, task keys and seeds. At21:43 UTC14 candidate tasks had
finished:10 successes and4 budget failures. No production runtime adoption or
completed measurement is claimed.

A third wave of26 newly admitted source batches is prepared and eligible on both
sandboxes:2491 LFM graphs and2487 Mellum graphs. It is separate from the already
adopted waves above; no third-wave exposure is claimed. Source preparation,
conversion, transport, classification and eligibility passed. Further Flint and
Qwen publications continue through the same pipeline. A transient GitHub DNS
failure stopped admission; the recovered worker retries network failures and
preserves exact unrelated engineering edits during data-only publication.

The full comparison and selection evidence is bound in
`/home/werg/data/dgxops/flint-full48-teacher-selection-root-20261021.json`.
The actual 48-case observation is preserved separately with its SHA-256. Flint
remains the active provisional generator because it supplies useful verified
positives and complementary examples; the result does not establish a quality
upgrade over Qwen.

Shared runtime fixes in `f458386e5c6dc4a0690e341ad2520ffa17b6605b` include all
required rendering assets in future actor compositions and keep parent-only
supervised readiness controls out of whole-graph RL. Full repository checks,
focused real optimizer/resume checks and the final quick check passed. Exact
rendering assets were restored on both active deployments. Mellum's corrected
policy operation actually advanced28181 to28182 with its original captured
on-policy episodes; LFM's R22 production continuation advanced past25037.
Neither statement is a claim of whole-recipe completion or measured capability
lift.
