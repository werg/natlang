const joinIds = rows => rows.length ? rows.map(row => row.id).join('; ') : 'none';

export function selectionContract({ count, direction, measureName, tieRule = 'ascending complete item ID' }) {
  if (!Number.isInteger(count) || count < 1)
    throw new Error('selection cardinality must be a positive integer');
  if (!['asc', 'desc'].includes(direction)) throw new Error('selection direction must be asc or desc');
  if (typeof measureName !== 'string' || !measureName.trim()) throw new Error('selection measure name is required');
  const cardinality = count === 1 ? 'exactly one eligible item when any qualify' : `up to ${count} eligible items`;
  const order = direction === 'asc' ? 'ascending' : 'descending';
  const countInstruction = count === 1
    ? 'Select exactly one eligible item when at least one qualifies; if none qualify, return none.'
    : `Select the first ${count} eligible items in the stated ranking when at least ${count} qualify. If fewer qualify, select all of them; if none qualify, return none.`;
  return {
    cardinality,
    select: rows => rows.slice(0, count),
    format: joinIds,
    formatMeasure: rows => rows.length ? rows.map(row => String(row.metric)).join('; ') : 'none',
    instruction: `${countInstruction} Rank by ${measureName} in ${order} order; resolve equal measures by ${tieRule}. Do not add further tied items.`,
    selectionFormat: count === 1
      ? `Exactly one complete item ID when any item qualifies, chosen by ${measureName} in ${order} order; equal measures use ${tieRule}. Do not include additional tied items. Use none when no item qualifies.`
      : `Up to ${count} complete item IDs: take the first ${count} eligible items in ${order} ${measureName} order when enough qualify, or all qualifying items when fewer do. Equal measures use ${tieRule}. Separate IDs with exactly semicolon and one space. Use none when no item qualifies.`,
    measureFormat: count === 1
      ? `${measureName} for the selected item as digits; use none when no item qualifies.`
      : `${measureName} values corresponding to the selected IDs in the same order, as digits separated by exactly semicolon and one space; do not sum them. Use none when no item qualifies.`,
  };
}

/** Build wording from task-authored authority metadata without inferring its status. */
export function authorizationContract({ action, noAction = 'no_action', requestId, requirement, scope }) {
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
