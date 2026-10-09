// Source "gate": gate report JSON files a run wrote; only the fields the watch entry names are extracted.
import type { Host } from '../host.js';
import type { GateReport, WatchEntry } from '../types.js';
import { failure, makeReading, type Reading } from './reading.js';

type Scalar = string | number | boolean | null;
const dig = (value: unknown, path: string): unknown => path.split('.').reduce<unknown>((at, key) =>
  at !== null && typeof at === 'object' ? (at as Record<string, unknown>)[key] : undefined, value);
const scalar = (value: unknown): Scalar => value === undefined ? null : typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' || value === null ? value : JSON.stringify(value);

export async function collectGates(host: Host, entry: WatchEntry): Promise<{ readings: Reading[], gates: GateReport[] }> {
  const readings: Reading[] = [], gates: GateReport[] = [];
  for (const path of entry.gate_paths) {
    const file = await host.readText(path);
    const subject = `${entry.run_id}/${path.split('/').slice(-2).join('/')}`;
    if (!file) { readings.push(failure(host, 'gate', subject, `the gate report ${path} does not exist (yet)`)); continue; }
    try {
      const json = JSON.parse(file.text) as unknown;
      const fields = Object.fromEntries(entry.gate_fields.filter(name => dig(json, name) !== undefined).map(name => [name, scalar(dig(json, name))]));
      const schema = String(dig(json, 'schema') ?? '');
      gates.push({ path, schema, fields });
      readings.push(makeReading(host, 'gate', subject, true, JSON.stringify({ path, schema, fields }, null, 2)));
    } catch (error) { readings.push(failure(host, 'gate', subject, `the gate report ${path} is not JSON: ${(error as Error).message}`)); }
  }
  return { readings, gates };
}

/** Whether a gate field reports failure: a false pass-field, a true fail-field, or a status that says failed. */
export function fieldFailed(name: string, value: Scalar): boolean {
  if (typeof value === 'string') return /fail|error|reject/i.test(value);
  if (typeof value !== 'boolean') return false;
  return /fail|error|reject|violat/i.test(name) ? value : !value;
}

export const gateFailures = (gates: GateReport[]): { path: string, field: string, value: Scalar }[] =>
  gates.flatMap(gate => Object.entries(gate.fields).filter(([name, value]) => fieldFailed(name, value)).map(([field, value]) => ({ path: gate.path, field, value })));
