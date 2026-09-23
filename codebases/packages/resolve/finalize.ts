export default function finalize(request: PackageRequest, locks: Lock[], selected: string): Resolution {
const lock = locks.find(row => row.id === selected);
if (!lock) return { status: 'invalid-selection',
  lock: { id: '', root: request.name, engine: request.engine, packages: [] },
  alternatives: locks.length, detail: 'selected ID was not offered by the exact solver' };
return { status: 'resolved', lock, alternatives: locks.length, detail: '' };
}
