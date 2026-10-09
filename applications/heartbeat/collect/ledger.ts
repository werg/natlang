// Source "ledger": `memory_ledger.py status` (claims, headroom).
import type { Host } from '../host.js';
import { failure, makeReading, type Reading } from './reading.js';

export type LedgerClaimStatus = { class: string, budget_gb: number, used_gb: number, peak_gb: number, outstanding_gb: number,
  hold_budget?: boolean, command: string };
export type LedgerStatus = { available_gb: number, headroom_gb: number, claims: Record<string, LedgerClaimStatus> };

export async function collectLedger(host: Host, repo: string): Promise<{ reading: Reading, status: LedgerStatus | null }> {
  const result = await host.exec(['python3', `${repo}/scripts/memory_ledger.py`, 'status'], { timeoutMs: 60_000, cwd: repo });
  if (!result.ok) return { reading: failure(host, 'ledger', 'status', result.stderr || result.stdout), status: null };
  try {
    const status = JSON.parse(result.stdout) as LedgerStatus;
    if (typeof status.headroom_gb !== 'number' || typeof status.claims !== 'object') throw new Error('the status has no headroom_gb and claims');
    return { reading: makeReading(host, 'ledger', 'status', true, result.stdout.trim()), status };
  } catch (error) {
    return { reading: failure(host, 'ledger', 'status', `unreadable ledger status: ${(error as Error).message}`), status: null };
  }
}
