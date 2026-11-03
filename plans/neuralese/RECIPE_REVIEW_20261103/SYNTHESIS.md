# LFM3 recipe review synthesis (2026-11-03)

Sources: `DATA_MIX_DIVERSITY.md`, `NEURALESE_CAPABILITY_COVERAGE.md`,
`TUNING_METALEARNING_RL.md` (independent sampled audits of the actual
production inputs; recipe `ops/earned179-joint-r38-a4-window64-1103`).

## Findings

1. **Two scripted generators dominate.** The `port_record_calls` wrapper
   (one instruction "Do what item.request asks…", 91% identical masked
   targets, research half mislabelled `natlang` at 4.72× weight) carries about 41% of
   loss. big2 template communication (86% of producers emit one identical
   program; 7-visit quotas on 14k-token rows) carries about 37% of loss and 65% of
   compute. About 138 genuine natlang teaching families get 5.6% of loss; genuine
   multi-node Neuralese graphs about 2%; real coding work about 2.6%.
2. **The intended behaviours are barely present.** Excluding the template:
   `iterateOn` 1.3%, directory reducers 3.4%, awaited `nl` 2.8%, `decide`
   0.22% (some malformed), skills 0.08%, stored guidance/decision-readout
   graphs 0. Producer→consumer depth ≥2 in about 170 rows. Luna codebases reach
   training as about 100 rows. Semantic delegation is 9% of categorized teaching
   versus the intended 35%; structural iteration 3% versus 15%.
3. **Only two Neuralese operators train** (marker rows and a rank-8 embedding
   adapter). Decision readout, stored guidance, soft states, LoRA
   bank/mixtures, import channel and metalearner have code but no data or
   training. Interface/backbone LRs are 1e-4/1e-6 against the committed 1e-3/2e-5;
   the reason needs verifying before any change.
4. **Formulaic data beyond the known ticket review.** About 1.6k keyword/regex
   substitutes for semantic judgment in other families; D16/D18 copy a 14-flag
   failure-label vector (80–97% duplicates); D12 trains only a closing summary
   behind a 59k-token transcript.
5. **RL is formulaic and not running.** 97.6% of the 61k tasks wrap a single
   dataset item in a root function; 99.1% are single calls. The 4-task × 4-sample
   selector starts new families at their easiest level, so rounds return 20/20
   with no outcome spread. This lineage has 0 RL rounds (`mandatory_sft_only`).
6. **Tuning and metalearning are outside the recipe and have never run on LFM3.**
   The recipe has one stage; stage gates check only completion and numerics, not
   task behaviour. Program supply for tuning is one structural shape.
7. **Realized throughput.** About 300 graphs/hour means a full pass takes about 160 days;
   recent updates were 88% big2 communication.

## Decisions

- Keep the DGX learner running until the corrected mixture is prepared (the GPU
  is otherwise idle), then cut over immediately through the normal
  checkpoint-preserving continuation. Exposure already earned stays recorded.
- Mixture correction (next SFT window):
  - downweight the wrapper to one row per task, out of the natlang family;
  - cap big2 at 1–2 visits and per template;
  - withdraw D16/D18 and collapse D12;
  - re-select the pending focus from diverse graph sources;
  - admit rest27 pairs, recovered sources and Luna codebases;
  - run the judgment-call regex review over all families.
- Start interleaved JOLT+ RL as a declared recipe stage, with a corrected
  selector (16–32 tasks × 6–8 samples, median starting level, spread-driven
  difficulty) and Neuralese-relay/typed/`iterateOn` task additions.
- Make tuning and metalearner stage kinds with task-behaviour promotion gates;
  run a bounded tuning pilot on cloud GPUs on the current parent.
- Promotion everywhere is by paired protected-development outcomes, never
  steps or loss.
