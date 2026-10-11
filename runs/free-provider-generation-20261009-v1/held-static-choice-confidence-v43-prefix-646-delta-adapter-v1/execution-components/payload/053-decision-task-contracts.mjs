import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { sha } from './common.mjs';

const repositoryRoot = resolve(fileURLToPath(new URL('../../../', import.meta.url)));
export const DECISION_TASK_CONTRACT_PATH = resolve(repositoryRoot, 'training/decision_task_contracts.json');
const contractBytes = readFileSync(DECISION_TASK_CONTRACT_PATH);
const contractDocument = JSON.parse(contractBytes.toString('utf8'));
if (contractDocument.schema !== 'natlang.decision-task-contracts/1' ||
    !contractDocument.contracts || typeof contractDocument.contracts !== 'object')
  throw new Error('invalid_shared_decision_task_contracts');
export const DECISION_TASK_CONTRACT_SHA256 = sha(contractBytes);

function validateCriteria(criteria, labels, where) {
  if (!criteria || Array.isArray(criteria) || typeof criteria !== 'object' ||
      Object.keys(criteria).length !== labels.length || labels.some(label =>
        !Object.hasOwn(criteria, label) || typeof criteria[label] !== 'string' || !criteria[label].trim()))
    throw new Error(`${where}: criteria keys must exactly match declared labels and have nonempty text`);
  return criteria;
}

/** Select explicit source criteria first, otherwise use the declared shared family taxonomy. */
export function decisionCriteria({ family, source, kind, labels, explicitCriteria, where = 'decision case' }) {
  if (!Array.isArray(labels) || !labels.length || labels.some(label => typeof label !== 'string' || !label))
    throw new Error(`${where}: labels must be nonempty strings`);
  if (explicitCriteria !== undefined && explicitCriteria !== null)
    return { criteria: validateCriteria(explicitCriteria, labels, where), provenance: {
      source: 'explicit_source_criteria', contract_path: DECISION_TASK_CONTRACT_PATH,
      contract_sha256: DECISION_TASK_CONTRACT_SHA256, family_contract_applied: false,
    } };
  const contract = contractDocument.contracts[family];
  if (!contract) return { criteria: Object.fromEntries(labels.map(label => [label, label])), provenance: null };
  if (contract.source !== source || contract.kind !== kind ||
      JSON.stringify(contract.label_order) !== JSON.stringify(labels))
    throw new Error(`${where}: source, kind, or labels do not match shared family contract`);
  return { criteria: validateCriteria(contract.criteria, labels, where), provenance: {
    source: 'shared_family_task_contract', contract_path: DECISION_TASK_CONTRACT_PATH,
    contract_sha256: DECISION_TASK_CONTRACT_SHA256, family_contract_applied: true,
    taxonomy_provenance: contract.provenance,
  } };
}
