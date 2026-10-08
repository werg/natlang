import { caseFrom } from './semantic-iterate-reducers-v18-novel-data.mjs';

/**
 * Turn task-authored authority scope into aligned rule, result, and pass guidance.
 * This helper states the task's rule; it does not read evidence or derive authority status.
 */
export function authorityWording({ action, noAction = 'no_action', requestId, requirement, scope }) {
  if (typeof action !== 'string' || !action.trim()) throw new Error('authorized action is required');
  if (typeof noAction !== 'string' || !noAction.trim()) throw new Error('no-action result is required');
  if (typeof requestId !== 'string' || !requestId.trim()) throw new Error('authorization request ID is required');
  if (typeof requirement !== 'string' || !requirement.trim()) throw new Error('authority requirement is required');
  if (!['request', 'each_selected_item'].includes(scope))
    throw new Error('authority scope must be request or each_selected_item');

  const authorityPhrase = /^(?:a|an|the)\b/i.test(requirement) ? requirement : `the ${requirement}`;
  const scopePhrase = scope === 'request'
    ? `for request ${requestId}`
    : `for each selected item in request ${requestId}`;
  const requirementText = `${authorityPhrase} ${scopePhrase}`;
  const scopeGuidance = scope === 'request'
    ? `The authority is scoped to request ${requestId}; do not add a separate item-specific authorization requirement.`
    : `The authority must cover every selected item; a request-level record alone does not establish item-specific authorization.`;
  return {
    requirementText,
    ruleText: `Decision mapping: if no item qualifies, use ${noAction}; if the nonempty eligible selection has ${requirementText} recorded, use ${action}; otherwise use hold. ${scopeGuidance}`,
    decisionFormat: `Use ${action} only when the eligible selection is nonempty and ${requirementText} is recorded; otherwise use hold. Use ${noAction} only when no item qualifies. Return one bare literal: ${action}, hold, ${noAction}.`,
    passConstraint: `Keep the evidence-derived selection and measure. Determine whether this source records the exact authority prerequisite: ${requirementText}. ${scopeGuidance} Authority affects only the decision; an empty eligible selection remains ${noAction}.`,
  };
}

/** Build a reducer case from a base eligibility/ranking rule and explicit authority metadata. */
export function caseFromWithAuthority(spec) {
  if (!spec || typeof spec !== 'object') throw new Error('case spec is required');
  if (typeof spec.selectionInstruction !== 'string' || !spec.selectionInstruction.trim())
    throw new Error(`${spec.slug}: selection instruction is required with authority metadata`);
  if (typeof spec.ruleText !== 'string' || !spec.ruleText.trim())
    throw new Error(`${spec.slug}: base rule text is required with authority metadata`);
  if (/selection cardinality:|decision mapping:/i.test(spec.ruleText))
    throw new Error(`${spec.slug}: provide only the base rule; selection and authority mapping are added by this builder`);

  const metadata = spec.authorizationRule;
  if (!metadata || typeof metadata !== 'object') throw new Error(`${spec.slug}: authorizationRule metadata is required`);
  if (typeof metadata.requirement !== 'string' ||
      !spec.ruleText.toLowerCase().includes(metadata.requirement.toLowerCase()))
    throw new Error(`${spec.slug}: authorizationRule requirement must match the named prerequisite in the base rule`);

  const wording = authorityWording({ action: spec.approvedAction, noAction: spec.noAction,
    requestId: spec.requestId, requirement: metadata.requirement, scope: metadata.scope });
  const row = caseFrom({ ...spec,
    ruleText: `${spec.ruleText} Selection cardinality: ${spec.selectionInstruction} ${wording.ruleText}`,
    decisionFormat: wording.decisionFormat,
  });
  row.passes = row.passes.map((pass, index) => index === row.passes.length - 1
    ? { ...pass, constraint: wording.passConstraint } : pass);
  return row;
}
