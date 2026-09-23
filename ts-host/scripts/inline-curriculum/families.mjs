// The family registry. `weight` scales the number of shapes per build toward the plan's domain shares.
import { abductionTest, entailmentException, proofVerifier } from './logic.mjs';
import { childSufficiency, cohortPolicy, contractDiagnosis, folderCriteria, moduleDiscovery, pagedLateRow, reviewEach } from './applications.mjs';
import { counterexampleRevision, idempotentRetry, inlineTypeRepair, lateBinding, liveInventory, parallelLabels, policyAfterMeasure } from './followup.mjs';
import { folioBatch, folioEntailment, prontoProof, prontoSearch } from './sources.mjs';
import { followUpDue, namedVersusInline, receiptTotals, releaseGate, triageUnion } from './inline.mjs';
import { childOptOut, eventRetry, loopRewrite } from './failure.mjs';
import { frontierSearch, greenhouseControl, lateArgmax } from './investigate.mjs';
import { kqaQuestion } from './kqapro.mjs';
import { textworldIterate, textworldQuest } from './textworld.mjs';
import { anliBatch, commaqaQuestion, entailmentPremises, proofwriterQuestion } from './sources-ai2.mjs';
import { iterateSchedule, routeReplan } from './actor.mjs';
import { dynamicSnapshot, multihopQualifier, policyCandidates } from './relational.mjs';

export const FAMILIES = {
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
  inline_type_repair: { build: inlineTypeRepair, weight: 2 },
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
  iterate_frontier: { build: frontierSearch, weight: 1.5 },
  relational_late_argmax: { build: lateArgmax, weight: 1.5 },
  actor_greenhouse: { build: greenhouseControl, weight: 2 },
  folio_entailment: { build: folioEntailment, weight: 3, source: 'folio' },
  folio_batch: { build: folioBatch, weight: 2, source: 'folio' },
  prontoqa_proof: { build: prontoProof, weight: 2, source: 'prontoqa' },
  prontoqa_search: { build: prontoSearch, weight: 2, source: 'prontoqa' },
  kqapro_question: { build: kqaQuestion, weight: 2, source: 'kqapro' },
  proofwriter_question: { build: proofwriterQuestion, weight: 2, source: 'proofwriter' },
  entailment_premises: { build: entailmentPremises, weight: 2, source: 'entailmentbank' },
  anli_batch: { build: anliBatch, weight: 2, source: 'anli' },
  commaqa_question: { build: commaqaQuestion, weight: 2, source: 'commaqa' },
  textworld_quest: { build: textworldQuest, weight: 2, source: 'textworld' },
  textworld_iterate: { build: textworldIterate, weight: 2, source: 'textworld' },
};
