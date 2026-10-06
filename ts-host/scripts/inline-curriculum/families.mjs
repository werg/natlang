import {decisionSkillCatalog, decisionExtractChain} from './decision-rich.mjs';
import { isDeepStrictEqual } from 'node:util';
import { decisionLambdas } from './decision-lambdas.mjs';
// The family registry. `weight` scales the number of shapes per build toward the plan's domain shares.
import { abductionTest, entailmentException, proofVerifier } from './logic.mjs';
import { childSufficiency, cohortPolicy, contractDiagnosis, folderCriteria, moduleDiscovery, pagedLateRow, reviewEach } from './applications.mjs';
import { counterexampleRevision, idempotentRetry, lateBinding, liveInventory, parallelLabels, policyAfterMeasure } from './followup.mjs';
import { folioBatch, folioEntailment, prontoProof, prontoSearch } from './sources.mjs';
import { followUpDue, namedVersusInline, receiptTotals, releaseGate, triageUnion } from './inline.mjs';
import { childOptOut, eventRetry, loopRewrite, recursionRewrite, reducerApply } from './failure.mjs';
import { frontierSearch, greenhouseControl, lateArgmax } from './investigate.mjs';
import { kqaQuestion } from './kqapro.mjs';
import { textworldIterate, textworldQuest } from './textworld.mjs';
import { alfworldTask, scienceworldTask } from './scienceworld.mjs';
import { authoringCapturedPolicy, authoringInlineReview, authoringIterate, authoringNamedHelper, authoringStructuredExtract } from './authoring.mjs';
import { anliBatch, commaqaNumeric, commaqaQuestion, entailmentPremises, proofwriterQuestion } from './sources-ai2.mjs';
import { iterateSchedule, routeReplan } from './actor.mjs';
import { dynamicSnapshot, multihopQualifier, policyCandidates } from './relational.mjs';
import { identifySourceCase, hintFor, hinted } from './lib.mjs';
import { composedHelpers, composedProcess } from './composed.mjs';
import { labeledJudgments } from './labeled.mjs';
import { folderTriageInline, folderIndexInline, folderTriage, folderIndex, folderMixed, folderEdit, folderFind, folderExtract } from './folder-families.mjs';
import { recurrenceGraphs, recurrenceInlineGraphs } from './recurrence.mjs';
import { chatReply, constrainedRewrite, constrainedWriting } from './writing.mjs';
import { crossSourceFolders } from './cross-source-folders.mjs';
import { workbenchApplicants, workbenchReviews, workbenchTickets } from './workbench.mjs';
import { datasetWorkbench } from './dataset-workbench.mjs';
import { translationDesk } from './translation-desk.mjs';
import { claimsDesk } from './claims-desk.mjs';
import { contractDesk } from './contract-desk.mjs';
import { editStream } from './edit-stream.mjs';
import { knowledgeEvidence, knowledgeResearch } from './knowledge-desk.mjs';
import { digestDesk, repoAnswer } from './bgkit-desk.mjs';
import { peopleChain, peopleLookup } from './people-desk.mjs';
import { citanceSummary, memoryAnswer, storyAnswer, storyChoice, webResearch } from './teacher-desk.mjs';

export const FAMILIES = {
  decision_skill_catalog: {build: decisionSkillCatalog, weight: 1, source: 'authored-bounded-decisions-v1'},
  decision_extract_chain: {build: decisionExtractChain, weight: 1, source: 'authored-bounded-decisions-v1'},
  decision_support: { build: (seed,index,split) => decisionLambdas(seed,index,split,'support'), weight: 1, source: 'authored-decision-worlds-v2' },
  decision_evidence: { build: (seed,index,split) => decisionLambdas(seed,index,split,'evidence'), weight: 1, source: 'authored-decision-worlds-v2' },
  decision_patch: { build: (seed,index,split) => decisionLambdas(seed,index,split,'patch'), weight: 1, source: 'authored-decision-worlds-v2' },
  decision_skills: { build: (seed,index,split) => decisionLambdas(seed,index,split,'skills'), weight: 1, source: 'authored-decision-worlds-v2' },

  cross_source_folders: { build: crossSourceFolders, weight: 1, externalData: true, source: 'semantic-cross-products' },
  recurrence_graphs: { build: recurrenceGraphs, weight: 1 },
  recurrence_inline_graphs: { build: recurrenceInlineGraphs, weight: 1 },
  logic_entailment_exception: { build: entailmentException, weight: 1.5 },
  logic_proof_verifier: { build: proofVerifier, weight: 1 },
  logic_abduction_test: { build: abductionTest, weight: 1 },
  relational_multihop_qualifier: { build: multihopQualifier, weight: 1 },
  relational_policy_inline: { build: policyCandidates, weight: 1 },
  relational_dynamic_snapshot: { build: dynamicSnapshot, weight: 1 },
  actor_route_replan: { build: routeReplan, weight: 2 },
  iterate_schedule_repair: { build: iterateSchedule, weight: 1 },
  inline_cohort_policy: { build: cohortPolicy, weight: 1 },
  inline_review_each: { build: reviewEach, weight: 1 },
  paged_late_row: { build: pagedLateRow, weight: 1 },
  contract_diagnosis: { build: contractDiagnosis, weight: 2 },
  folder_criteria_reducer: { build: folderCriteria, weight: 2 },
  child_sufficiency: { build: childSufficiency, weight: 1 },
  scoped_module_discovery: { build: moduleDiscovery, weight: 1 },
  policy_after_measure: { build: policyAfterMeasure, weight: 1 },
  counterexample_revision: { build: counterexampleRevision, weight: 1 },
  parallel_labels: { build: parallelLabels, weight: 1 },
  idempotent_retry: { build: idempotentRetry, weight: 3 },
  live_inventory: { build: liveInventory, weight: 1.5 },
  inline_late_binding: { build: lateBinding, weight: 1.5 },
  inline_multi_capture: { build: releaseGate, weight: 1 },
  inline_union_target: { build: triageUnion, weight: 1 },
  stateful_dates: { build: followUpDue, weight: 1 },
  inline_structured_extract: { build: receiptTotals, weight: 1 },
  named_versus_inline: { build: namedVersusInline, weight: 1 },
  loop_rewrite: { build: loopRewrite, weight: 2 },
  child_opt_out: { build: childOptOut, weight: 1 },
  event_retry: { build: eventRetry, weight: 3 },
  recursion_rewrite: { build: recursionRewrite, weight: 2 },
  reducer_apply: { build: reducerApply, weight: 2 },
  iterate_frontier: { build: frontierSearch, weight: 1.5 },
  relational_late_argmax: { build: lateArgmax, weight: 1.5 },
  actor_greenhouse: { build: greenhouseControl, weight: 2 },
  composed_helpers: { build: composedHelpers, weight: 2 },
  composed_process: { build: composedProcess, weight: 2 },
  // Generated per seed from labeled datasets in the cache (acquire.mjs): sms_spam, sst2, ag_news, emotion, banking77, clinc_oos.
  labeled_judgments: { build: labeledJudgments, weight: 3, externalData: true },
  folder_triage: { build: folderTriage, weight: 1, externalData: true },
  folder_triage_inline: { build: folderTriageInline, weight: 1, externalData: true },
  folder_index_inline: { build: folderIndexInline, weight: 1, externalData: true },
  folder_index: { build: folderIndex, weight: 1, externalData: true },
  folder_mixed: { build: folderMixed, weight: 1, externalData: true },
  folder_edit: { build: folderEdit, weight: 1, externalData: true },
  folder_find: { build: folderFind, weight: 1, externalData: true },
  folder_extract: { build: folderExtract, weight: 1, externalData: true },
  folio_entailment: { build: folioEntailment, weight: 3, source: 'folio' },
  folio_batch: { build: folioBatch, weight: 2, source: 'folio' },
  prontoqa_proof: { build: prontoProof, weight: 2, source: 'prontoqa' },
  prontoqa_search: { build: prontoSearch, weight: 2, source: 'prontoqa' },
  kqapro_question: { build: kqaQuestion, weight: 2, source: 'kqapro' },
  proofwriter_question: { build: proofwriterQuestion, weight: 2, source: 'proofwriter' },
  entailment_premises: { build: entailmentPremises, weight: 2, source: 'entailmentbank' },
  anli_batch: { build: anliBatch, weight: 2, source: 'anli' },
  commaqa_question: { build: commaqaQuestion, weight: 2, source: 'commaqa' },
  commaqa_numeric: { build: commaqaNumeric, weight: 2, source: 'commaqa' },
  textworld_quest: { build: textworldQuest, weight: 2, source: 'textworld' },
  textworld_iterate: { build: textworldIterate, weight: 2, source: 'textworld' },
  scienceworld_task: { build: scienceworldTask, weight: 2, source: 'scienceworld' },
  alfworld_task: { build: alfworldTask, weight: 2, source: 'alfworld' },
  // Chat and writing as natlang domains: open text admitted by code-checked constraints (oracle level "constraints").
  constrained_writing: { build: constrainedWriting, weight: 2 },
  constrained_rewrite: { build: constrainedRewrite, weight: 2 },
  chat_reply: { build: chatReply, weight: 2 },
  workbench_tickets: { build: workbenchTickets, weight: 3 },
  workbench_reviews: { build: workbenchReviews, weight: 3 },
  workbench_applicants: { build: workbenchApplicants, weight: 3 },
  dataset_workbench: { build: datasetWorkbench, weight: 4 },
  claims_desk: { build: claimsDesk, weight: 2 },
  contract_desk: { build: contractDesk, weight: 2 },
  edit_stream: { build: editStream, weight: 2 },
  knowledge_evidence: { build: knowledgeEvidence, weight: 2, externalData: true, contentIdentity: true, source: 'hotpotqa' },
  knowledge_research: { build: knowledgeResearch, weight: 2, externalData: true, contentIdentity: true, source: 'hotpotqa' },
  digest_desk: { build: digestDesk, weight: 2, externalData: true, contentIdentity: true, source: 'bgkit' },
  repo_answer: { build: repoAnswer, weight: 2, externalData: true, contentIdentity: true, source: 'bgkit' },
  people_lookup: { build: peopleLookup, weight: 1, externalData: true, contentIdentity: true, source: 'schnitzeljagd-synth-people' },
  people_chain: { build: peopleChain, weight: 2, externalData: true, contentIdentity: true, source: 'schnitzeljagd-synth-people' },
  // Teacher collection only (generation.collection 'teacher'): minimal references, intermediate answers unknown.
  web_research: { build: webResearch, weight: 2, externalData: true, contentIdentity: true, collection: 'teacher', source: 'bgkit-web' },
  memory_answer: { build: memoryAnswer, weight: 2, externalData: true, contentIdentity: true, collection: 'teacher', source: 'bgkit-memory-qa' },
  story_choice: { build: storyChoice, weight: 1, externalData: true, contentIdentity: true, collection: 'teacher', source: 'quality' },
  story_answer: { build: storyAnswer, weight: 1, externalData: true, contentIdentity: true, collection: 'teacher', source: 'narrativeqa' },
  citance_summary: { build: citanceSummary, weight: 1, externalData: true, contentIdentity: true, collection: 'teacher', source: 'schnitzeljagd-citances' },
  // Demonstrations only (static replay): no per-item translation oracle for teacher outputs yet.
  translation_desk: { build: translationDesk, weight: 1, demonstration: true },
  // The TypeScript authoring track (curriculum.track "authoring").
  authoring_inline_review: { build: authoringInlineReview, weight: 1, track: 'authoring' },
  authoring_iterate: { build: authoringIterate, weight: 1, track: 'authoring' },
  authoring_named_helper: { build: authoringNamedHelper, weight: 1, track: 'authoring' },
  authoring_structured_extract: { build: authoringStructuredExtract, weight: 1, track: 'authoring' },
  authoring_captured_policy: { build: authoringCapturedPolicy, weight: 1, track: 'authoring' },
};

/**
 * The cases a build makes: `shapes` shapes per family (scaled by its weight) from index `start`, with ids, split
 * groups and pair groups tagged by seed for generated families and by source for sourced ones. `hints` adds each
 * technique case's hinted twin.
 */
export function buildRecords({ seed, shapes, start = 0, families, split = 'train', hints = true }) {
  const records = [];
  for (const name of families) {
    const family = FAMILIES[name];
    const count = Math.max(1, Math.round(shapes * (family.weight ?? 1)));
    for (let index = start; index < start + count; index++) for (const record of family.build(seed, index, split)) {
      if (family.contentIdentity) identifySourceCase(record);
      // Generated problems are distinct per seed; a source's problems are its own (story, world) groups across shards.
      const tag = family.source ? `${family.source}` : `s${seed}`;
      record.id = record.id.replace('inline-curriculum:', `inline-curriculum:${tag}:`);
      record.source_ids = record.dataset_records ?? [record.id];
      if (!family.source) {
        record.split = split;
        record.curriculum.split_group = `s${seed}:${record.curriculum.split_group}`;
        record.source_groups = record.dataset_records?.map(id => `${record.dataset}:${id}`) ?? [record.curriculum.split_group];
        if (record.curriculum.pair_group) record.curriculum.pair_group = `s${seed}:${record.curriculum.pair_group}`;
      }
      // A source adapter samples a large dataset, so two indexes can land on the same problem: keep the first.
      // Generated families must never repeat an id.
      const existing = records.find(other => other.id === record.id);
      if (existing) {
        if (family.source) {
          const oracle = value => ({ expected: value.semantics.expected,
            files: value.semantics.expected_files, answer_oracle: value.semantics.oracle,
            files_oracle: value.semantics.files_oracle });
          if (!isDeepStrictEqual(oracle(existing), oracle(record)))
            throw new Error(`conflicting source oracle for case id ${record.id}`);
          continue;
        }
        throw new Error(`duplicate case id ${record.id}`);
      }
      records.push(record);
      // A case that requires iterateOn or per-item nl judgments also gets a twin whose instructions end with an
      // explicit hint (and the family's sketch). Admission strips the hint from the twin's trajectory, so it trains
      // the technique without being asked; with the unhinted run it makes a same-request preference pair (pairs.mjs).
      const hint = hintFor(record.curriculum);
      if (hints && hint && record.curriculum.track !== 'authoring') records.push(hinted(record, hint));
    }
  }
  return records;
}
