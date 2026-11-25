import {isDeepStrictEqual} from 'node:util';

const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const positiveChecks = checks => object(checks) && Object.keys(checks).length > 0 &&
  Object.values(checks).every(value => value === true);
const acceptedReview = review => object(review) && review.schema === 'natlang.live-workflow-review/1' &&
  Boolean(review.model) && object(review.input) && Object.keys(review.input).length > 0 &&
  review.verdict?.accepted === true && [review.exchange, ...(review.exchanges ?? [])].some(
    exchange => object(exchange) && object(exchange.request) && object(exchange.reply));

/** Same bound whole-goal verdict as live_teacher_sources.semantic_goal_completed.
 * Call only after reading hashed evidence and verifying any completed reassessment.
 */
export function semanticGoalCompleted(interaction, proofs, verification, evidence = [], reassessmentVerified = false) {
  if (verification.status !== 'passed' || !positiveChecks(verification.checks)) return false;
  const goal = interaction?.goal;
  if (verification.reassessment != null) return reassessmentVerified && proofs.length === 1 &&
    proofs[0].schema === 'natlang.live-outcome-reassessment/1' &&
    proofs[0].task_id === verification.task_id && proofs[0].goal === goal &&
    proofs[0].passed === true && positiveChecks(proofs[0].checks);
  const reviews = evidence.filter(item => item.semanticReview != null).map(item => item.semanticReview);
  const requirements = verification.requirements ?? {};
  if (!object(requirements)) return false;
  if (Object.hasOwn(requirements, 'semantic_review') && typeof requirements.semantic_review !== 'boolean') return false;
  const requiredReview = requirements.semantic_review ?? Object.hasOwn(verification.checks, 'independentSemanticReview');
  const reviewed = reviews.length === 1 && acceptedReview(reviews[0]);
  if (requiredReview && !reviewed) return false;
  const scoped = proofs.filter(proof => ['goal', 'caseId', 'contextId'].some(key => Object.hasOwn(proof, key)));
  if (!scoped.length) return !interaction;
  const world = verification.world ?? {};
  const scenario = String(verification.task_id ?? '').split(':').at(-1);
  const caseId = world.case_id ?? (scenario.startsWith('open-goal-') ? scenario.slice('open-goal-'.length) : null);
  if (!goal || !world.context_id || !caseId || verification.root_call_ids?.length !== 1) return false;
  const exact = scoped.filter(proof => proof.goal === goal && proof.contextId === world.context_id &&
    proof.caseId === caseId && (!interaction?.entry || isDeepStrictEqual(proof.entry, interaction.entry)));
  if (exact.length !== 1 || exact[0].passed !== true) return false;
  const proof = exact[0];
  if (proof.semanticReviewRequired != null && typeof proof.semanticReviewRequired !== 'boolean') return false;
  if (proof.semanticReviewRequired === true && !reviewed || !positiveChecks(proof.checks)) return false;
  const dimensions = proof.outcomeDimensions;
  if (dimensions != null) {
    if (!object(dimensions) || !Object.keys(dimensions).length ||
        Object.values(dimensions).some(dimension => !object(dimension) || typeof dimension.passed !== 'boolean')) return false;
    const contract = proof.outcomeContract ?? {};
    if (!object(contract)) return false;
    for (const [name, declaration] of Object.entries(contract)) {
      if (!object(declaration) || Object.hasOwn(declaration, 'required') && typeof declaration.required !== 'boolean' ||
          (declaration.required ?? name !== 'incidentalValues') && !Object.hasOwn(dimensions, name)) return false;
    }
    for (const [name, dimension] of Object.entries(dimensions)) {
      const declaration = contract[name] ?? {};
      if (!object(declaration)) return false;
      const flags = [dimension, declaration].filter(role => Object.hasOwn(role, 'required')).map(role => role.required);
      if (flags.some(flag => typeof flag !== 'boolean') || new Set(flags).size > 1 ||
          (flags[0] ?? name !== 'incidentalValues') && dimension.passed !== true) return false;
    }
  }
  return true;
}
