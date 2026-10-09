// Source "unit": `systemctl --user show` for each watched unit, and the list of natlang units on the machine.
import type { Host } from '../host.js';
import type { UnitState } from '../types.js';
import { failure, makeReading, type Reading } from './reading.js';

const PROPERTIES = ['LoadState', 'ActiveState', 'SubState', 'Result', 'ExecMainStatus', 'ActiveEnterTimestamp'];

export function parseUnit(text: string): UnitState & { load_state: string } {
  const fields = Object.fromEntries(text.split('\n').filter(line => line.includes('=')).map(line => [line.slice(0, line.indexOf('=')), line.slice(line.indexOf('=') + 1).trim()]));
  const status = fields.ExecMainStatus;
  return { load_state: fields.LoadState ?? 'unknown', active_state: fields.ActiveState ?? 'unknown', sub_state: fields.SubState ?? 'unknown',
    result: fields.Result ?? 'unknown', exit_status: status !== undefined && /^-?\d+$/.test(status) ? Number(status) : null, started_at: fields.ActiveEnterTimestamp ?? '' };
}

export async function collectUnit(host: Host, unit: string): Promise<{ reading: Reading, state: UnitState | null }> {
  const result = await host.exec(['systemctl', '--user', 'show', unit, ...PROPERTIES.flatMap(name => ['-p', name])], { timeoutMs: 20_000 });
  if (!result.ok) return { reading: failure(host, 'unit', unit, result.stderr || result.stdout), state: null };
  const { load_state, ...state } = parseUnit(result.stdout);
  // A unit systemd no longer knows (a finished transient unit) is a fact about the run, not a failure to read.
  return { reading: makeReading(host, 'unit', unit, true, result.stdout.trim() + (load_state === 'not-found' ? '\n(systemd has no unit by this name)' : '')), state };
}

export type ListedUnit = { unit: string, active_state: string, sub_state: string };

/** Every natlang-* service systemd knows, running or not. */
export async function collectUnitList(host: Host): Promise<{ reading: Reading, units: ListedUnit[] }> {
  const result = await host.exec(['systemctl', '--user', 'list-units', 'natlang-*', '--all', '--plain', '--no-legend', '--type=service'], { timeoutMs: 20_000 });
  if (!result.ok) return { reading: failure(host, 'unit', 'list', result.stderr || result.stdout), units: [] };
  const units = result.stdout.split('\n').map(line => line.trim().split(/\s+/)).filter(parts => parts.length >= 4 && parts[0]!.endsWith('.service'))
    .map(parts => ({ unit: parts[0]!, active_state: parts[2]!, sub_state: parts[3]! }));
  return { reading: makeReading(host, 'unit', 'list', true, result.stdout.trim()), units };
}
