# Decision-model distillation with runtime-composed natlang lambdas

## Research checked 2026-10-05

Cloudflare exposes `@cf/cloudflare/clef` and `@cf/cloudflare/clef-flash` through
Workers AI. Both accept a state and typed questions; outputs are bounded choices,
yes/no probabilities or rubric scores. They cannot author arbitrary code or prose.
The current documented context is65,536tokens, with up to64questions per request.
Use the model-specific REST endpoint, not a chat-completion adapter.
Sources: [Clef](https://developers.cloudflare.com/workers-ai/models/clef/),
[Clef-flash](https://developers.cloudflare.com/workers-ai/models/clef-flash/).

The free allocation is **10,000neurons/day shared across the account**. Current
rates are21,818neurons/M input tokens for Clef and8,182 for flash: approximately
458k or1.22M input tokens if the entire allocation goes to that one model. They
are not two independent allowances. Questions/state/overhead consume input budget;
other Workers AI usage also consumes the account allocation. Reset is00:00UTC.
[Official pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/).

Concrete Jev applications that fit our program design:

| Observed application | Suitable natlang program / captured context |
| --- | --- |
| Skill ranking followed by a suitability check | Task + complete skill descriptions + execution feedback + already loaded skills; shortlist then reject unnecessary skills |
| Semantic line search | Query + candidate passages + exclusions + document structure; select spans, then copy exact bytes |
| Bounded value extraction | Document + requested role + candidate spans + normalization contract; choose a span, code owns formatting/arithmetic |
| Entity alignment | Two records + ontology + relationship evidence + merge policy; decide match/separate/review before exact tree edits |
| Hierarchical classification | Document + current taxonomy node + path history + competing sibling descriptions; construct the next question after choosing a branch |
| Workflow triage | Ticket/invoice/incident + policy + prior conversation + account/service facts; route, collect evidence, reassess |
| Code/agent trace verification | Requested behavior + patch or tool trace + contract + validation evidence; select accept/change/review |
| Games and simulation | Goal + legal actions + simulated consequences + recent history; pick, execute deterministic transition, reformulate |

These are grounded in primary examples, not claims that every demo is production
validated: [skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion),
[line search](https://docs.typesafe.ai/cookbooks/semantic_find),
[bounded extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook),
[entity alignment](https://docs.typesafe.ai/cookbooks/entity_alignment),
[hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification),
[community workflow implementations](https://github.com/kenhuangus/jev-usecases),
[game harness](https://github.com/christianmat/jev-pokemon).

A research study reports that supplying predicted consequences helps Jev when an
unassisted question would require internal simulation. Our inference: prepare
candidate outcomes/checker results in code and let bounded judgments compare
those outcomes, rather than expecting a decision model to generate missing plans.
[Paper](https://arxiv.org/html/2610.01834v1). That result is about Jev; it is not a
measured guarantee for Clef.

## Implemented first families

`decision_support`, `decision_evidence`, `decision_patch`, `decision_skills` are
registered in the existing curriculum family registry. The controller programs
are authored; they formulate questions at runtime, loop over actual files, and
run semantic child calls. Clef supplies verified decisions inside those programs.
We are not claiming Clef generates the entire program or a reasoning narrative.

The operative text-form pattern is:

```ts
const item = await file.readJson();
let history = histories[item.id];
let question = ['Primary decision:', task.objective, task.exceptions].join(' ');
let decision = await nl<Decision>`
  ${question} Use item, policy, history and catalog.
  Return only the selected catalog key.
`();
if (decision === task.reviewKey && extras[item.id]) {
  history += '\n' + extras[item.id];
  question = ['Follow-up decision:', task.objective,
    'Reassess after the new history; previous catalog choice was ' + decision +
    '. Do not retain a decision if new evidence changes it.'].join(' ');
  decision = await nl<Decision>`
    ${question} Use item, policy, history and catalog.
    Return only the selected catalog key.
  `();
}
```

The lambda refers to lexical scope by name. Arguments are **not** flattened into
instruction prose. Its child first executes `eval` to read these actual captured
values, so training contexts show the evidence before the decision. The provider
state has the same item/policy/history/catalog/question. Follow-up formulation
and updated history depend on the preceding child result, not a precomputed flat
list of unrelated questions. Option keys are permuted; catalog descriptions, not
opaque labels, define their meaning.

Choice maps to a finite union; Noul maps to a declared boolean decision with an
explicit threshold; Score maps to a judgment/rubric value, not an exact numeric
calculation. For a structured result, several bounded questions may populate its
fields, with schema validation and recorded probabilities. Arbitrary strings,
patches and rewrites need candidate generation or a generative model. Multiple
questions sharing state can be batched, but a dependent follow-up needs a later
request because one answer is not another question's input in the same request.

## Quality pilot and course changes

Wrangler4.105.0 OAuth login includes AI access. Live REST probe succeeded. No new
Worker deployment, additional key, GPU reservation or persistent service required.
The collector calls `wrangler auth token --json` into a private process pipe;
credentials never enter artifacts. Explicit environment credentials also work.
Requests omit chat-only/distillation parameters. Transport/rate-limit retries use
bounded exponential backoff with jitter and respect Retry-After.

Two small pilots ran. The first had6/16whole programs accepted. Review identified
an overly optimistic authored patch oracle: a missing format implementation fact
still warranted review even when a breaking change was authorized. We corrected
that oracle, emphasized policy precedence, and preserved the first run as a held
diagnostic rather than silently replacing its evidence.

The revised pilot had **5/16accepted programs,85approved native decisions**:
4skill-selection programs and1patch-review program.68requests used29,604input
tokens, approximately330neurons.53flash calls agreed with the oracle46times;
15targeted Clef escalations agreed9times. Escalations were selected hard examples,
so these rates are not a head-to-head model comparison.11whole programs stayed
held:6ended with a wrong selected choice and5with a correct choice below the
probability/margin gate. We stop a failed case early; this is not an exhaustive
per-item accuracy evaluation.

Current gate: independent authored-world agreement, chosen probability>=0.8,
margin>=0.2, then actual runtime replay and native decision admission. Probability
is not a substitute for correctness. Full response probabilities, requests,
expected oracle labels, escalations and rejected cases are retained. Training
receives only admitted native traces; oracle labels remain outside visible inputs.
Authored action-plan supervision is explicitly marked, never teacher reasoning.

Initial contexts without a child scope-read step were incomplete for supervision;
those early oracle traces are superseded and not published as training. Corrected
source pool uses stable fixture identities across seeds/catalog permutations.
The16cases reuse a small set of authored facts: they are a mechanism pilot, not
16independent real-world tasks. Do not manufacture held-out novelty with new IDs.
The two pilot collectors used a fixed trace root seed6131, independent of case
seed; case/source hashes preserve real inputs. Future collector derives trace seed
from each case shape. That trace seed did not change the scripted decisions.

## Next scaling steps

1. Expand skill selection against our real, versioned skill index. Include none,
   near misses, already-loaded skills, changed descriptions, competing tools and
   multiple necessary skills; validate on unseen tasks/catalog fixtures.
2. Split policy-heavy decisions into bounded subquestions when beneficial:
   establish source admissibility/currentness, then judge a claim from selected
   evidence. Generate the second lambda from the first result. Keep both steps
   and the exact code joining them; do not hard-code the teacher's final label.
3. Use bounded extraction to obtain rich structured/text returns by selecting and
   copying exact source spans. This complements crisp-only decision calls and
   improves meaningful recurrence coverage without fabricating free-form prose.
4. Cross independently sampled requests, policies, history and candidate catalogs
   from licensed existing datasets. Preserve original source IDs and protection
   groups across every combination. Pair changed policies/evidence while keeping
   the underlying item fixed; require a justified changed outcome.
5. Add code/constraint-backed candidate optimization and semantic games with real
   transition/checker results. `iterateOn` updates state, objectives and prompts;
   Clef ranks generated candidates rather than inventing unchecked edits.
6. Add visual/frontend decisions with Clef's supported image inputs only after
   artifact capture and graders are in place.
7. Before unattended daily scheduling, add persistent account-wide neuron budget,
   date rollover, quota-exhaustion handling and resumable request cache. The current
   bounded pilot is not that scheduler. Do not spend two model-specific quotas.

CLI:

```sh
node ts-host/scripts/inline-curriculum/build-static-lambda-corpus.mjs \
  --seed 6132 --shapes 2 \
  --families decision_support,decision_evidence,decision_patch,decision_skills \
  --authored-plans --supervise-reference-answers --out NEW_SOURCE_DIRECTORY
node ts-host/scripts/inline-curriculum/collect-clef-lambdas.mjs \
  --cases NEW_SOURCE_DIRECTORY/cases.jsonl --out NEW_PILOT_DIRECTORY \
  --max-tokens 60000 --min-probability 0.8
```

Source/runtime correctness is verified separately from live teacher judgment.
Native-to-neuralese conversion and global source/split/producer closure remain
required before joining the training builder.

## 2026-10-05 reviewed recurrence expansion and policy-stage transition

Expanded bounded Clef families: skill-catalog selection and nested evidence extraction.
Reviewed packet contains 78 train programs / 52 held programs, built from 12 train
versus 8 held authored fixtures (variants are not independent semantic worlds).
`clef-rich-reviewed-cohort-20261005-v1` admits 1120 train / 747 held native
records, 670 writers, 677 producer edges, depth 5 and branching 3. Exact oracle,
source provenance, stable fixture split and producer closure all gate admission.
Six immutable registry snapshots preserve sources, teacher receipts (including
failed partial collection), admitted records, eval ordering, old-data reconversion
and task-evaluation evidence. No provider receipts themselves become SFT targets.

Conversion/5 matches canonical and runtime-rendered values using observed parentage.
Old static-lambda expansion now recovers 319 writers / 319 edges rather than the
old 28 writers; it remains a candidate subject to its existing quality policies.
Ambiguous equal outputs under the same parent remain exact text; never fabricate
a producer link. Inline instruction literals remain a later curriculum step.

Task evaluation at joint-stage step256: all five conditional return arms scored
0/4; autonomous crisp-runtime runs scored 0/4 skill tasks and 0/4 extraction tasks.
Wrong schemas/values, failure to inspect scope and unnecessary skill selection
are policy failures, not a proven channel-only problem. This checkpoint descends
from the original 350M port backbone with frozen inherited last-four-layer LoRA;
it is not the best broad ordinary-SFT student. Do not advertise capability gains
from teacher-prefix CE alone.

Course change: gracefully stop joint-learning-v2 after its step512 probe (own
written CE .958745 versus shuffled 1.096745; 24 readers, 70.83% own better).
Retain full optimizer/RNG checkpoint and best state; the original 2048-step stage
is superseded, not falsely marked complete. Next mixed stage uses the new
reviewed corpus, five-layer recurrence, trainable policy adapters and sequential
ordinary-text SFT replay. A new stage gets a fresh optimizer explicitly; unchanged
stages still resume all optimizer/RNG state. `--soft-init` carries matching soft
parameters forward; full deployment export validates pinned parent metadata.
Crisp replay restores structured write sources as typed objects and accumulates
gradients only after recurrent tapes are freed. Periodic crisp CE accompanies
written/shuffled controls; actual free-output task checks remain necessary.

Validation: 36 scoped TS tests passed; 9 Python state/render tests passed in the
pinned Torch/Muon container. New mixed GPU stage must be checked for memory fit
and convergence; zero prior errors does not establish its behavior.
