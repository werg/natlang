/** Exact, host-only provenance for a named one-item NL reducer and its scoped child. */
import { isDeepStrictEqual } from 'node:util';

type Dict = Record<string, unknown>;
type Capture = { name?: unknown; complete?: unknown; value?: unknown; reason?: unknown };
type LedgerRow = { invocation_id?: unknown; parent_invocation_id?: unknown; completion_status?: unknown;
  inline_instruction_site?: unknown; host_result?: unknown; host_inputs?: unknown };
type IdentityField = { child_input_pointer: string; completed_item_pointer: string;
  result_pointer?: string; comparison: 'exact' | 'unordered_unique_ids' };
type BindingSite = { owner_source_path: string; inline_child_site: { return_type: string }; frontier_input_pointer: string;
  next_state_pointer: string; prior_result_array_pointer: string; new_slot_pointer: string;
  prior_completed_array_pointer: string; next_completed_array_pointer: string; pending_array_pointer: string;
  consumed_input_selector: { kind: 'first_pending_item' | 'newly_completed_identity'; pointer: string };
  item_identity_fields: IdentityField[]; semantic_fields: string[];
  owned_pending_update: 'remove_exact_first_item' | 'remove_exact_selected_identity'; stable_state_fields: string[];
  witness_projection?: { schema: 'semantic-source-owner-witness-projection/1'; result_pointer: string;
    child_result_pointer: string; owner_witnesses_from: { document: 'scoped_child_input'; pointer: string };
    identity_fields: string[]; allowed_modes: string[] } | null };
export type ItemResultBindingSpec = { schema: 'semantic-actual-child-result-binding/1'; sites: BindingSite[] };
export type ItemResultBindingVerdict = { status: 'pass' | 'fail' | 'unverified_hold';
  site_results: { source: string; checked_steps: number; status: 'pass' | 'fail' | 'unverified_hold'; issues: string[] }[];
  training_admission: false };

const object = (value: unknown): value is Dict => value !== null && typeof value === 'object' && !Array.isArray(value);
const own = (value: unknown, key: string): unknown => object(value) && Object.hasOwn(value, key) ? value[key] : undefined;
function at(value: unknown, pointer: string): { found: boolean; value: unknown } {
  if (pointer === '') return { found: true, value };
  if (!pointer.startsWith('/')) return { found: false, value: undefined };
  let current = value;
  for (const escaped of pointer.slice(1).split('/')) {
    const key = escaped.replace(/~1/g, '/').replace(/~0/g, '~');
    if (Array.isArray(current) && /^(0|[1-9][0-9]*)$/.test(key)) current = current[Number(key)];
    else if (object(current) && Object.hasOwn(current, key)) current = current[key];
    else return { found: false, value: undefined };
    if (current === undefined) return { found: false, value: undefined };
  }
  return { found: true, value: current };
}
function captured(row: LedgerRow, name: string): { status: 'ok' | 'missing'; value?: unknown } {
  const found = Array.isArray(row.host_inputs) ? (row.host_inputs as Capture[]).find(input => input.name === name) : undefined;
  return found?.complete === true ? { status: 'ok', value: found.value } : { status: 'missing' };
}
function result(row: LedgerRow): { status: 'ok' | 'missing'; value?: unknown } {
  const found = row.host_result as Capture | undefined;
  return found?.complete === true ? { status: 'ok', value: found.value } : { status: 'missing' };
}
function sameIdentity(left: unknown, right: unknown, mode: IdentityField['comparison']): boolean {
  if (mode === 'exact') return isDeepStrictEqual(left, right);
  if (!Array.isArray(left) || !Array.isArray(right) || !left.every(x => typeof x === 'string') ||
      !right.every(x => typeof x === 'string')) return false;
  return new Set(left).size === left.length && new Set(right).size === right.length &&
    left.length === right.length && left.every(x => right.includes(x));
}
function prefixAndAppend(previous: unknown, next: unknown): boolean {
  return Array.isArray(previous) && Array.isArray(next) && next.length === previous.length + 1 &&
    previous.every((value, index) => isDeepStrictEqual(value, next[index]));
}
function sourceOf(row: LedgerRow): unknown { return own(row.host_result, 'definition_source') ??
  (Array.isArray(row.host_inputs) ? own(row.host_inputs[0], 'definition_source') : undefined); }
function returnTypeOf(row: LedgerRow): string {
  const site = row.inline_instruction_site;
  const returns = own(site, 'returns');
  return String(own(returns, 'natlang') ?? own(returns, 'text') ?? own(row.host_result, 'result_type') ?? '');
}
function projectedWitnesses(child: unknown[], owner: unknown[], identityFields: string[]): unknown[] | null {
  const all = [...child, ...owner];
  const keys: string[] = [];
  for (const witness of all) {
    const fields = identityFields.map(pointer => at(witness, pointer));
    if (fields.some(field => !field.found)) return null;
    keys.push(JSON.stringify(fields.map(field => field.value)));
  }
  return all.filter((_, index) => keys.indexOf(keys[index]!) === index);
}

/** This is diagnostic qualification, never an answer oracle or an admission decision. */
export function checkItemResultBinding(spec: ItemResultBindingSpec, ledger: readonly LedgerRow[]): ItemResultBindingVerdict {
  const site_results = spec.sites.map(site => {
    const parents = ledger.filter(row => sourceOf(row) === site.owner_source_path);
    const issues: string[] = [];
    let checked_steps = 0;
    let missing = false;
    for (const parent of parents) {
      const id = parent.invocation_id;
      if (typeof id !== 'string' || parent.completion_status !== 'done') { missing = true; continue; }
      const inputCapture = captured(parent, 'state');
      const outputCapture = result(parent);
      if (inputCapture.status !== 'ok' || outputCapture.status !== 'ok') { missing = true; continue; }
      const prior = at({ state: inputCapture.value }, site.frontier_input_pointer);
      const next = at(outputCapture.value, site.next_state_pointer);
      if (!prior.found || !next.found) { issues.push(`${id}: typed frontier missing`); continue; }
      const priorResults = at(prior.value, site.prior_result_array_pointer).value;
      const nextResults = at(next.value, site.new_slot_pointer).value;
      const priorCompleted = at(prior.value, site.prior_completed_array_pointer).value;
      const nextCompleted = at(next.value, site.next_completed_array_pointer).value;
      const priorPending = at(prior.value, site.pending_array_pointer).value;
      const nextPending = at(next.value, site.pending_array_pointer).value;
      if (!prefixAndAppend(priorResults, nextResults) || !prefixAndAppend(priorCompleted, nextCompleted) ||
          !Array.isArray(priorPending) || !Array.isArray(nextPending)) {
        issues.push(`${id}: result/completed frontier did not append one item`); continue;
      }
      if (!Array.isArray(priorResults) || !Array.isArray(nextResults) ||
          !Array.isArray(priorCompleted) || !Array.isArray(nextCompleted)) {
        issues.push(`${id}: result/completed frontier is not an array`); continue;
      }
      const childRows = ledger.filter(row => row.parent_invocation_id === id && row.completion_status === 'done' &&
        returnTypeOf(row).includes(site.inline_child_site.return_type));
      if (childRows.length !== 1) { issues.push(`${id}: expected one completed scoped child`); continue; }
      const child = childRows[0]!;
      const childInput = captured(child, 'input');
      const childResult = result(child);
      if (childInput.status !== 'ok' || childResult.status !== 'ok') { missing = true; continue; }
      const slot = nextResults[priorResults.length];
      const completed = nextCompleted[priorCompleted.length];
      const selected = site.consumed_input_selector.kind === 'first_pending_item' ? priorPending[0] : completed;
      for (const field of site.item_identity_fields) {
        const childValue = at(childInput.value, field.child_input_pointer);
        const completedValue = at(completed, field.completed_item_pointer);
        if (!childValue.found || !completedValue.found || !sameIdentity(childValue.value, completedValue.value, field.comparison))
          issues.push(`${id}: scoped input/completed item identity differs at ${field.child_input_pointer}`);
        if (field.result_pointer !== undefined) {
          const resultValue = at(childResult.value, field.result_pointer);
          if (!resultValue.found || !childValue.found || !sameIdentity(childValue.value, resultValue.value, field.comparison))
            issues.push(`${id}: scoped output identity differs at ${field.result_pointer}`);
        }
      }
      if (site.consumed_input_selector.kind === 'first_pending_item' && !isDeepStrictEqual(selected, completed))
        issues.push(`${id}: completed item is not first pending item`);
      if (site.owned_pending_update === 'remove_exact_first_item' &&
          !isDeepStrictEqual(nextPending, priorPending.slice(1))) issues.push(`${id}: pending did not drop first item`);
      if (site.owned_pending_update === 'remove_exact_selected_identity' &&
          (!priorPending.some(value => isDeepStrictEqual(value, selected)) ||
           !isDeepStrictEqual(nextPending, priorPending.filter(value => !isDeepStrictEqual(value, selected)))))
        issues.push(`${id}: pending did not remove selected identity`);
      for (const pointer of site.stable_state_fields)
        if (!isDeepStrictEqual(at(prior.value, pointer), at(next.value, pointer))) issues.push(`${id}: stable ${pointer} changed`);
      for (const field of site.semantic_fields) {
        if (field === 'witnesses' && site.witness_projection) continue;
        if (!isDeepStrictEqual(own(slot, field), own(childResult.value, field)))
          issues.push(`${id}: child ${field} was changed or bypassed`);
      }
      if (site.witness_projection) {
        const witnessSpec = site.witness_projection;
        const childWitnesses = at(childResult.value, witnessSpec.child_result_pointer).value;
        const slotWitnesses = at(slot, witnessSpec.result_pointer).value;
        const ownerWitnesses = at(childInput.value, witnessSpec.owner_witnesses_from.pointer).value;
        const union = Array.isArray(childWitnesses) && Array.isArray(ownerWitnesses) ?
          projectedWitnesses(childWitnesses, ownerWitnesses, witnessSpec.identity_fields) : null;
        if (!Array.isArray(childWitnesses) || !Array.isArray(slotWitnesses) || !Array.isArray(ownerWitnesses) ||
            !(isDeepStrictEqual(slotWitnesses, childWitnesses) || (union && isDeepStrictEqual(slotWitnesses, union))))
          issues.push(`${id}: witness projection changed child or owner evidence`);
      }
      checked_steps++;
    }
    const status: ItemResultBindingVerdict['status'] = issues.length ? 'fail' : missing || !checked_steps ? 'unverified_hold' : 'pass';
    return { source: site.owner_source_path, checked_steps, status, issues };
  });
  const status = site_results.some(item => item.status === 'fail') ? 'fail' :
    site_results.some(item => item.status === 'unverified_hold') ? 'unverified_hold' : 'pass';
  return { status, site_results, training_admission: false };
}
